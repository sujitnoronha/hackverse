from django.db import models

# Create your models here.
class analytics(models.Model):
    location = models.CharField(max_length=100,blank= False, null=False)
    time = models.DateTimeField(auto_now_add=True)
    peoplecount = models.IntegerField()
    socialdistancing = models.IntegerField()
    scenedetect = models.name = models.CharField(max_length=60, blank=True, null=True)
    sceneimage = models.ImageField(upload_to='sceneimage/',blank = True, null=True)

    def __str__(self):
        return f'{self.location}'



class person(models.Model):
    name = models.CharField(max_length=256)
    image = models.ImageField(upload_to="personimage/",blank=True, null=True)
    crimes = models.CharField(max_length=1000, blank =True, null=True)

    def __str__(self):
        return str(self.name)

class plocation(models.Model):
    name = models.ForeignKey('person', on_delete=models.CASCADE)
    location = models.CharField(max_length =100)
    time = models.DateTimeField(auto_now_add = True,blank=True, null=True)
    
    def __str__(self):
        return str(self.name)