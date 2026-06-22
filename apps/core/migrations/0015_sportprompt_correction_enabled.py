from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0014_sportprompt_taglines"),
    ]

    operations = [
        migrations.AddField(
            model_name="sportprompt",
            name="correction_enabled",
            field=models.BooleanField(default=False),
        ),
    ]
