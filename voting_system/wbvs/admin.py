from django.contrib import admin
from .models import User, Booth, Candidate, Voter, VotingList, Result, History

# Register your models here.
admin.site.register(User)
admin.site.register(Booth)
admin.site.register(Candidate)
admin.site.register(Voter)
admin.site.register(History)
admin.site.register(VotingList)
admin.site.register(Result)