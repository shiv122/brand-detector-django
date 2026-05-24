from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0009_seed_sport_prompts"),
    ]

    operations = [
        migrations.AddField(
            model_name="sportprompt",
            name="allowed_brands",
            field=models.JSONField(blank=True, default=list),
        ),
    ]
