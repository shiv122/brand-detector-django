from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0013_frame_ocr_job_id"),
    ]

    operations = [
        migrations.AddField(
            model_name="sportprompt",
            name="taglines",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
