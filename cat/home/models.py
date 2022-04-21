import django.utils.timezone as timezone

from django.db import models


# Create your models here.
class cat(models.Model):
    id = models.AutoField(primary_key=True)
    image = models.ImageField(upload_to="%Y%m%d/")
    predict = models.CharField(default="No Results", max_length=60, blank=False, null=False)
    create_time = models.DateTimeField(auto_now_add=True)
    flag = models.IntegerField(default=1, choices=((0, "False"), (1, "True")))

    class Meta:
        db_table = "cat"
