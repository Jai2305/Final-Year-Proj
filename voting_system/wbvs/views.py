from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse
from django import forms
import re
from django.contrib.auth.hashers import make_password

from .models import Result, User, Booth, Candidate, Voter, VotingList, History, Feedback

from django.contrib.auth.decorators import login_required
from django.contrib import messages
import uuid
from .fsave import fsave 
from .fsaveMatch1 import fsm


def fpsave(request) :
    fsave()
    return render(request,'recieve.html')

class RegisterForm(forms.Form):
    first_name = forms.CharField(label = "",widget= forms.TextInput(attrs={'placeholder':'First Name', 'class':'form-group form-control', 'autofocus type':'text'}), max_length=20)
    last_name = forms.CharField(label = "",widget= forms.TextInput(attrs={'placeholder':'Last Name', 'class':'form-group form-control', 'autofocus type':'text'}), max_length=20)
    email = forms.CharField(label = "",widget= forms.TextInput(attrs={'placeholder':'Adhaar No.', 'class':'form-group form-control', 'autofocus type':'text'}), max_length=12)
    booth = forms.CharField(label = "",widget= forms.TextInput(attrs={'placeholder':'Booth No.', 'class':'form-group form-control', 'autofocus type':'text'}), max_length=20)
    voteridno = forms.CharField(label = "",widget= forms.TextInput(attrs={'placeholder':'Voter ID', 'class':'form-group form-control', 'autofocus type':'text'}), max_length=10)
    #password = forms.CharField(label = "",widget=forms.PasswordInput(attrs={'placeholder':'Password', 'class':'form-group form-control', 'autofocus type':'text'}))
    #confirmation = forms.CharField(label = "",widget=forms.PasswordInput(attrs={'placeholder':'Confirm Password', 'class':'form-group form-control', 'autofocus type':'text'}))
    image = forms.ImageField(label = "")


class LoginForm(forms.Form):
    email = forms.CharField(label = "",widget= forms.TextInput(attrs={'placeholder':'Aadhar No.', 'class':'form-group form-control', 'autofocus type':'text'}), max_length=12)
    voteridno = forms.CharField(label = "",widget= forms.TextInput(attrs={'placeholder':'Voter ID.', 'class':'form-group form-control', 'autofocus type':'text'}), max_length=10)
    #password = forms.CharField(label = "",widget=forms.PasswordInput(attrs={'placeholder':'Password', 'class':'form-group form-control', 'autofocus type':'text'}))


class PasswordChangeForm(forms.Form):
    new_password = forms.CharField(label = "",widget=forms.PasswordInput(attrs={'placeholder':'New Password', 'class':'form-group form-control', 'autofocus type':'text'}))
    confirmation = forms.CharField(label = "",widget=forms.PasswordInput(attrs={'placeholder':'Confirm New Password', 'class':'form-group form-control', 'autofocus type':'text'}))


class AdminForm1(forms.Form):
    title = forms.CharField(label = "",widget= forms.TextInput(attrs={'placeholder':'Title', 'class':'form-group form-control', 'autofocus type':'text'}))
    description = forms.CharField(label = "",widget= forms.Textarea(attrs={'placeholder':'Description',"rows":5, "cols":20, 'class':'form-group form-control', 'autofocus type':'text'}))


class AdminForm2(forms.Form):
    name = forms.CharField(label = "",max_length=16,widget= forms.TextInput(attrs={'placeholder':'Candidate Name', 'class':'form-group form-control', 'autofocus type':'text'}))
    des = forms.CharField(label = "",widget= forms.Textarea(attrs={'placeholder':'Candidate Description',"rows":3, 'class':'form-group form-control', 'autofocus type':'text'}))
    cand_im = forms.ImageField(label = "")


class VoterForm1(forms.Form):
    boothID = forms.CharField(label = "",widget= forms.TextInput(attrs={'placeholder':'Booth ID', 'class':'form-group form-control', 'autofocus type':'text'}))


class FeedbackForm(forms.Form):
    subject = forms.CharField(label = "",max_length=16,widget= forms.TextInput(attrs={'placeholder':'Subject', 'class':'form-group form-control', 'autofocus type':'text'}))
    feedback = forms.CharField(label = "",widget= forms.Textarea(attrs={'placeholder':'Feedback',"rows":5, 'class':'form-group form-control', 'autofocus type':'text'}))


def generate_boothID():
    var = uuid.uuid4().hex[:10].upper()
    var = var + "@booth"
    return var


def index(request):
    return render(request, "wbvs/index.html", {
        "feedback_form" : FeedbackForm(),
    })


@login_required(login_url='login')
def homepage(request):
    user = str(request.user)
    print(user)
    if user.find("@user") != -1:
        print("Found!")
        return render(request, "wbvs/homepage.html", {
        "voterform1" : VoterForm1(),
        "adminform1" : AdminForm1(),
        "boothID" : generate_boothID(),
        "feedback_form" : FeedbackForm(),
    })
    else:
        print("Not found!")
        return render(request, "wbvs/homepageAdmin.html", {
        "voterform1" : VoterForm1(),
        "adminform1" : AdminForm1(),
        "boothID" : generate_boothID(),
        "feedback_form" : FeedbackForm(),
    })
    

@login_required(login_url='login')
def homepageAdmin(request):
    print(request.user)
    

def authenticate_custom(username,voteridno):
    try:
        user = User.objects.get(username=username,voteridno=voteridno)
    except User.DoesNotExist:
        user = None
    return user

def login_view(request):
    if request.method == "POST":

        # Attempt to sign user in
        email = request.POST["email"]
        voteridno = request.POST["voteridno"]
        #password = request.POST["password"]
        #username = re.findall('(\S+)@', email)
        username = str(email + "@user")
        user = authenticate_custom(username,voteridno)
        print(user)
        print(User.objects.filter(is_superuser=True))

        # Check if authentication successful
        if user is not None:
            login(request, user)
            return render(request, "wbvs/homepage.html", {
                "success_login" : True,
                "voterform1" : VoterForm1(),
                "adminform1" : AdminForm1(),
                "boothID" : generate_boothID(),
                "feedback_form" : FeedbackForm()
            })
        else:
            return render(request, "wbvs/login.html", {
                "message": "Invalid Aadhar no. or Booth No.",
                "login_form" : LoginForm(request.POST),
                "feedback_form" : FeedbackForm(),
            })
    else:
        return render(request, "wbvs/login.html", {
            "login_form" : LoginForm(),
            "feedback_form" : FeedbackForm(),
        })

def logout_view(request):
    logout(request)
    return render(request, "wbvs/index.html", {
        "success_logout" : True,
        "feedback_form" : FeedbackForm(),
    })


def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST, request.FILES)
        if form.is_valid():
            noi = fsave()
            first_name = form.cleaned_data["first_name"]
            last_name = form.cleaned_data["last_name"]
            email = form.cleaned_data["email"]
            image = form.cleaned_data["image"]
            booth = form.cleaned_data["booth"]
            voteridno = form.cleaned_data["voteridno"]
            #username = re.findall('(\S+)@', email)
            username = str(email + "@user")

            # Ensure password matches confirmation
            #password = form.cleaned_data["password"]
            #confirmation = form.cleaned_data["confirmation"]
            #if password != confirmation:
            #    return render(request, "wbvs/register.html", {
            #        "message": "Passwords must match.",
            #        "register_form" : RegisterForm(request.POST),
            #        "feedback_form" : FeedbackForm(),
            #    })

            # Attempt to create new user user.image.url
            try:
                user = User.objects.create_user(email = email, first_name = first_name, last_name = last_name, username = username, image = image, booth=booth,noi=noi,voteridno=voteridno)
                user.save()
            except IntegrityError:
                return render(request, "wbvs/register.html", {
                    "message": "Aadhar no. not.",
                    "register_form" : RegisterForm(request.POST),
                    "feedback_form" : FeedbackForm(),
                })
            login(request, user)
            return render(request, "wbvs/homepage.html", {
                "success_register" : True,
                "voterform1" : VoterForm1(),
                "adminform1" : AdminForm1(),
                "boothID" : generate_boothID(),
                "feedback_form" : FeedbackForm(),
            })
    else:
        return render(request, "wbvs/register.html", {
            "register_form" : RegisterForm(),
            "feedback_form" : FeedbackForm(),
        })


@login_required(login_url='login')
def booth_creator(request):
    var = generate_boothID()
    return render(request, "wbvs/homepage.html", {
        "boothID" : var,
        "adminform1" : AdminForm1(),
        "voterform1" : VoterForm1(),
        "feedback_form" : FeedbackForm(),
    })

 
@login_required(login_url='login')
def adminform1(request, Id):
    user_var = User.objects.get(pk = request.user.id)
    if request.method == "POST":
        data = AdminForm1(request.POST)
        if data.is_valid():
            ti = data.cleaned_data["title"]
            des = data.cleaned_data["description"]
            Booth.objects.create(boothID = Id, active = True, admin = request.user)
            Booth.objects.filter(boothID = Id, active = True).update(title = ti, admin = user_var, description = des)
            History.objects.create(user = request.user, boothID = Id, role = "Voting Admin", result_declared = False, result = "", active = True, voting_status = "NA")
            return HttpResponseRedirect(reverse("adminpage2", args=(Id, )))
        else:
            return render(request, "wbvs/homepage.html",{
                "voterform1" : VoterForm1(),
                "adminform1" : AdminForm1(),
                "boothID" : generate_boothID(),
                "feedback_form" : FeedbackForm(),
            })

@login_required(login_url='login')
def adminpage2(request, Id):
    candidate_var = Candidate.objects.filter(boothID = Id).all()
    if request.method == "POST":
        form = AdminForm2(request.POST, request.FILES)
        if form.is_valid():
            var = uuid.uuid4().hex[:6].upper()
            var = var + "@candidate"

            name = form.cleaned_data["name"]
            des = form.cleaned_data["des"]
            cand_image = form.cleaned_data["cand_im"]
            
            Candidate.objects.create(boothID = Id, candidateID = var, candidateName = name, description = des, image = cand_image)
            return HttpResponseRedirect(reverse("adminpage2", args=(Id,)))
    else:
        return render(request, "wbvs/adminpage2.html", {
            "boothID" : Id,
            "candidate_list" : candidate_var,
            "adminform2" : AdminForm2(),
            "feedback_form" : FeedbackForm(),
        })


@login_required(login_url='login')
def adminpage3(request, Id):
    voter_list = Voter.objects.filter(boothID = Id).all()
    if request.method == "POST":
        voter_var = User.objects.get(pk = int(request.POST["foo"]))
        request_rply = 1
        Voter.objects.filter(voterID = voter_var).update(allowed = 1)
        if (request_rply == -1):
            Voter.objects.filter(voterID = voter_var, boothID = Id).update(voting_status = "Rejected")
            History.objects.filter(user = voter_var, boothID = Id).update(voting_status = "Rejected")
        elif (request_rply == 1):
            History.objects.filter(user = voter_var, boothID = Id).update(voting_status = "Not Voted")
        return HttpResponseRedirect(reverse("adminpage3", args=(Id, )))

    candidate_var = Candidate.objects.filter(boothID = Id).all()
    if len(candidate_var) < 2:
        return render(request, "wbvs/adminpage2.html", {
            "cnadidate_count_warning" : True,
            "boothID" : Id,
            "candidate_list" : candidate_var,
            "adminform2" : AdminForm2(),
            "feedback_form" : FeedbackForm(),
        })

    return render(request, "wbvs/adminpage3.html", {
        "boothID" : Id,
        "voter_list" : voter_list,
        "total_requests" : len(Voter.objects.filter(boothID = Id).all()),
        "approved_requests" : len(Voter.objects.filter(boothID = Id , allowed = 1).all()),
        "denied_requests" : len(Voter.objects.filter(boothID = Id , allowed = 1).all()),
        "voted_votes" : len(Voter.objects.filter(boothID = Id , allowed = 1, voting_status = "Voted").all()),
        "pending_votes" : len(Voter.objects.filter(boothID = Id , allowed = 1, voting_status = "Not Voted").all()),
        "feedback_form" : FeedbackForm(),
    })


@login_required(login_url='login')
def boothcheck(request):
    if request.method == "POST":
        data = VoterForm1(request.POST)
        if data.is_valid():
            allowed = True
            boo = data.cleaned_data["boothID"]
            candidate_var = Candidate.objects.filter(boothID = boo).all()
            try:
                res = 1 
                print(str(request.user.id))
                noi=User.objects.get(pk = request.user.id).noi
                print(noi)
                print(boo)
                print(request.user.booth)
                if boo != request.user.booth+"@booth" : raise
                if noi == "null" :raise
                #if not noi : noi= "null"
                #print(noi)
                res = fsm(noi)
                print(res)
                booth = Booth.objects.get(boothID = boo, active = True)
                if res == 1 : raise
            except:
                return render(request, "wbvs/homepage.html", {
                    "failed_booth_find" : True,
                    "voterform1" : VoterForm1(),
                    "adminform1" : AdminForm1(),
                    "boothID" : generate_boothID(),
                    "feedback_form" : FeedbackForm(),
                })
            History.objects.create(user = request.user, boothID = boo, role = "Voter", result_declared = False, result = "", active = True, voting_status = "W8ing_4_req")
            return render(request, "wbvs/waiting.html", {
                "allowed" : allowed,
                "boothID" : boo,
                "feedback_form" : FeedbackForm(),
            })

    return render(request, "wbvs/homepage.html", {
        "voterform1" : VoterForm1(),
        "adminform1" : AdminForm1(),
        "boothID" : generate_boothID(),
        "feedback_form" : FeedbackForm(),
    })


@login_required(login_url='login')
def access(request, Id):
    Voter.objects.create(boothID = Id, voterID = request.user, allowed = 1, voting_status = "Not Voted")
    return HttpResponseRedirect(reverse("waiting", args=(Id, )))


@login_required(login_url='login')
def waiting(request, Id):
    var = Voter.objects.get(boothID = Id, voterID = request.user)
    var.allowed = 1
    allowed = True
    not_allowed = False
    
    if var.allowed == 1:
        allowed = True
    if var.allowed == -1:
        return render(request, "wbvs/homepage.html", {
            "permission_denied" : True,
            "voterform1" : VoterForm1(),
            "adminform1" : AdminForm1(),
            "boothID" : generate_boothID(),
            "feedback_form" : FeedbackForm(),
        })
    
    return render(request, "wbvs/waiting.html", {
        "allowed" : allowed,
        "boothID" : Id,
        "feedback_form" : FeedbackForm(),
    })


@login_required(login_url='login')
def voterpage3(request , Id):
    
    #waiting(request, Id)
    if request.method == "POST":
        access(request,Id)
        voter_var = User.objects.get(pk = request.user.id)
        Voter.objects.filter(boothID = Id, voterID = request.user).update(voting_status = "Voted")
        VotingList.objects.create(boothID = Id, candidateID = request.POST["candidate_selected"], voterID = voter_var)
        History.objects.filter(boothID = Id, role = "Voter").update(user = request.user, voting_status = "Voted")
        return render(request, "wbvs/homepage.html", {
            "success_vote" : True,
            "voterform1" : VoterForm1(),
            "adminform1" : AdminForm1(),
            "boothID" : generate_boothID(),
            "feedback_form" : FeedbackForm(),
        })
    
    candidate_var = Candidate.objects.filter(boothID = Id).all()
    return render(request, "wbvs/voterpage3.html", {
        "boothId" : Id,
        "candidate_list" : candidate_var,
        "feedback_form" : FeedbackForm(),
    })


def give_winner(Id):
    entries = Result.objects.filter(boothID = Id).all()
    max = -1
    winner = ""
    for entry in entries:
        if entry.vote_count > max:
            winner = entry.candidateID
            max = entry.vote_count
    
    winner = Candidate.objects.get(candidateID = winner).candidateName
    return winner


@login_required(login_url='login')
def calculate(request, Id):
    Booth.objects.filter(boothID = Id).update(active = False)
    entries = VotingList.objects.filter(boothID = Id).all()
    candidate_list = Candidate.objects.filter(boothID = Id).all()
    for candidate in candidate_list:
        cand_var = Result.objects.create(boothID = Id, candidateID = candidate.candidateID, vote_count = 0)
        count = 0
        for entry in entries:
            if candidate.candidateID == entry.candidateID:
                count += 1
        Result.objects.filter(pk = cand_var.id).update(vote_count = count)
    History.objects.filter(boothID = Id).update(result_declared = True, result = f"{give_winner(Id)} WON", active = False)
    return HttpResponseRedirect(reverse("view_result", args=(Id, )))


@login_required(login_url='login')
def view_result(request, Id):
    booth = Booth.objects.get(boothID = Id)
    result_pg_availaible = not(booth.active)

    entries = Result.objects.filter(boothID = Id).all()
    data = dict()
    for i in entries:
        data[i.candidateID] = i.vote_count

    winner = give_winner(Id)

    return render(request, "wbvs/result.html", {
        "winner" : winner,
        "show_result" : result_pg_availaible,
        "candidates_data" : data,
        "boothID" : Id,
        "feedback_form" : FeedbackForm(),
    })


@login_required(login_url='login')
def history(request):
    data = History.objects.filter(user = request.user).all()
    return render(request, "wbvs/history.html", {
        "data" : data,
        "feedback_form" : FeedbackForm(),
    })


@login_required(login_url='login')
def view_details(request):
    return render(request, "wbvs/view_details.html", {
        "feedback_form" : FeedbackForm(),
    })


@login_required(login_url='login')
def edit_password(request):
    if request.method == "POST":
        new_password = request.POST["new_password"]
        confirmation = request.POST["confirmation"]

        if new_password != confirmation:
            return render(request, "wbvs/edit_password.html", {
                "message": "Passwords must match.",
                "passwd_form" : PasswordChangeForm(request.POST),
                "feedback_form" : FeedbackForm(),
            })
        User.objects.filter(username = request.user.username).update(password = make_password(new_password))
        return HttpResponseRedirect(reverse("index"))

    return render(request, "wbvs/edit_password.html", {
        "passwd_form" : PasswordChangeForm(),
        "feedback_form" : FeedbackForm(),
    })

@login_required(login_url='login')
def feedback(request):
    if request.method == "POST":
        data = FeedbackForm(request.POST)
        if data.is_valid():
            sub = request.POST["subject"]
            feed = request.POST["feedback"]
            Feedback.objects.create(user = request.user, subject = sub, feedback = feed)
            return render(request, "wbvs/homepage.html", {
                "voterform1" : VoterForm1(),
                "adminform1" : AdminForm1(),
                "boothID" : generate_boothID(),
                "feedback_form" : FeedbackForm(),
                "feedback_msg" : True
            })
        else:
            return render(request, "wbvs/homepage.html", {
                "voterform1" : VoterForm1(),
                "adminform1" : AdminForm1(),
                "boothID" : generate_boothID(),
                "feedback_form" : FeedbackForm(),
                "feedback_error_msg" : True
            })
    return render(request, "wbvs/homepage.html", {
        "voterform1" : VoterForm1(),
        "adminform1" : AdminForm1(),
        "boothID" : generate_boothID(),
        "feedback_form" : FeedbackForm(),
    })