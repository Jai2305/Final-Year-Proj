from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db.models.base import Model
from django.utils.timezone import now

 
# Create your models here.
class User(AbstractUser):
    image = models.ImageField(upload_to = "images/user", blank=True)
    noi = models.CharField(max_length=16)
    booth = models.CharField(max_length=20)
    def __str__(self):
        return self.username


class Booth(models.Model):
    boothID = models.CharField(max_length=16)
    title = models.CharField(max_length=16)
    active = models.BooleanField()
    admin = models.ForeignKey(User, on_delete=models.CASCADE, related_name="user_as_admin")
    description = models.TextField(blank=True)
    def __str__(self):
        return self.boothID


class Candidate(models.Model):
    boothID = models.CharField(max_length=16)
    candidateID = models.CharField(max_length=16)
    candidateName = models.CharField(max_length=16)
    description = models.CharField(max_length=200)
    image = models.ImageField(upload_to = "images/candidates", blank=True)
    def __str__(self):
        return self.candidateID 


class Voter(models.Model):
    boothID = models.CharField(max_length=16)
    voterID = models.ForeignKey(User, on_delete=models.CASCADE, related_name="voter_in_voterlist")
    allowed = models.IntegerField(default=0)
    voting_status = models.CharField(max_length=16, null=True)
    fingerprint = models.CharField(max_length=16)


class VotingList(models.Model):
    boothID = models.CharField(max_length=16)
    candidateID = models.CharField(max_length=16)
    voterID = models.ForeignKey(User, on_delete=models.CASCADE, related_name="voter_voted")


class Result(models.Model):
    boothID = models.CharField(max_length=16)
    candidateID = models.CharField(max_length=16)
    vote_count = models.PositiveIntegerField(default=0)


class History(models.Model):
    active = models.BooleanField()
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="history_user")
    boothID = models.CharField(max_length=16)
    role = models.CharField(max_length=16)
    result_declared = models.BooleanField()
    result = models.CharField(max_length=16)
    voting_status = models.CharField(max_length=16, default = "Not Voted")

class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="feedback_by")
    subject = models.CharField(max_length=16)
    feedback =  models.CharField(max_length=200)