"""Drop everything from the OCR RQ queue.

Default behavior clears pending, started, failed, deferred, and scheduled
registries — i.e. a hard reset. Pass --only to keep some buckets.

Examples:
    python manage.py clear_ocr_queue                    # nuke all buckets
    python manage.py clear_ocr_queue --only pending     # just pending jobs
    python manage.py clear_ocr_queue --only failed      # just the failed bin
    python manage.py clear_ocr_queue --yes              # skip confirmation
    python manage.py clear_ocr_queue --queue some_other_q
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError


BUCKETS = ("pending", "started", "failed", "deferred", "scheduled")


class Command(BaseCommand):
    help = "Clear the OCR RQ queue (pending + started + failed + deferred + scheduled)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--queue",
            default="ocr",
            help="Queue name (default: ocr).",
        )
        parser.add_argument(
            "--only",
            nargs="+",
            choices=BUCKETS,
            help=(
                "Buckets to clear. Defaults to all of: "
                + ", ".join(BUCKETS)
                + "."
            ),
        )
        parser.add_argument(
            "--yes",
            "-y",
            action="store_true",
            help="Skip the interactive confirmation prompt.",
        )

    def handle(self, *args, **opts):
        try:
            import django_rq
        except ImportError as exc:
            raise CommandError(f"django_rq is not installed: {exc}") from exc

        queue_name = opts["queue"]
        targets = tuple(opts["only"]) if opts["only"] else BUCKETS

        try:
            queue = django_rq.get_queue(queue_name)
        except Exception as exc:  # noqa: BLE001 — any connect/lookup failure
            raise CommandError(
                f"Could not connect to queue '{queue_name}': {exc}. "
                "Is Redis running and REDIS_URL set?"
            ) from exc

        counts_before = self._count_buckets(queue)
        total_before = sum(counts_before.values())
        self._print_summary("Before", counts_before)

        if total_before == 0:
            self.stdout.write(self.style.SUCCESS("Queue is already empty."))
            return

        if not opts["yes"]:
            scope = "ALL buckets" if targets == BUCKETS else f"only: {', '.join(targets)}"
            self.stdout.write(
                f"About to clear {scope} from queue '{queue_name}'."
            )
            confirm = input("Type 'yes' to proceed: ").strip().lower()
            if confirm != "yes":
                self.stdout.write("Aborted.")
                return

        removed = 0
        if "pending" in targets:
            removed += self._clear_pending(queue)
        if "started" in targets:
            removed += self._clear_registry(queue.started_job_registry, "started")
        if "failed" in targets:
            removed += self._clear_registry(queue.failed_job_registry, "failed")
        if "deferred" in targets:
            removed += self._clear_registry(queue.deferred_job_registry, "deferred")
        if "scheduled" in targets:
            removed += self._clear_registry(queue.scheduled_job_registry, "scheduled")

        counts_after = self._count_buckets(queue)
        self._print_summary("After", counts_after)
        self.stdout.write(
            self.style.SUCCESS(f"Removed {removed} job(s) from '{queue_name}'.")
        )

    @staticmethod
    def _count_buckets(queue) -> dict[str, int]:
        return {
            "pending": queue.count,
            "started": len(queue.started_job_registry),
            "failed": len(queue.failed_job_registry),
            "deferred": len(queue.deferred_job_registry),
            "scheduled": len(queue.scheduled_job_registry),
        }

    def _print_summary(self, label: str, counts: dict[str, int]) -> None:
        rendered = ", ".join(f"{k}={v}" for k, v in counts.items())
        total = sum(counts.values())
        self.stdout.write(f"{label}: total={total} ({rendered})")

    def _clear_pending(self, queue) -> int:
        n = queue.count
        queue.empty()
        return n

    def _clear_registry(self, registry, label: str) -> int:
        try:
            job_ids = registry.get_job_ids()
        except Exception as exc:  # noqa: BLE001
            self.stderr.write(
                self.style.WARNING(f"Could not enumerate {label} jobs: {exc}")
            )
            return 0

        removed = 0
        from rq.job import Job
        from rq.exceptions import NoSuchJobError

        connection = registry.connection
        for jid in job_ids:
            try:
                job = Job.fetch(jid, connection=connection)
                job.delete()
                removed += 1
            except NoSuchJobError:
                # Job already gone — strip the dangling registry entry.
                try:
                    registry.remove(jid)
                except Exception:  # noqa: BLE001
                    pass
            except Exception as exc:  # noqa: BLE001 — best-effort per job
                self.stderr.write(
                    self.style.WARNING(f"Skip {label} job {jid}: {exc}")
                )
        return removed
