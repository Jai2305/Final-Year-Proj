from django import template
from wbvs.models import Result, User, Booth, Candidate, Voter, VotingList, History
register = template.Library()

@register.simple_tag
def show_image(candidate_id):
    return Candidate.objects.get(candidateID = candidate_id).image.url
    
@register.simple_tag
def show_name(candidate_id):
    return Candidate.objects.get(candidateID = candidate_id).candidateName

@register.simple_tag
def get_voter_name(id):
    return User.objects.get(pk = id).first_name + ' ' + User.objects.get(pk = id).last_name


@register.simple_tag
def get_voter_mail(id):
    return User.objects.get(pk = id).email
