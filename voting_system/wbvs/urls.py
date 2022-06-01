from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.index, name="index"),
    path("fpsave",views.fpsave,name="fpsave"),
    path("homepage", views.homepage, name="homepage"),
    path("login", views.login_view, name="login"),
    path("logout", views.logout_view, name="logout"),
    path("register", views.register, name="register"),

    path("boothCreator", views.booth_creator, name="booth_creator"),
    path("<str:Id>/adminform1", views.adminform1, name="adminform1"),
    path("<str:Id>/adminpage2", views.adminpage2, name="adminpage2"),
    path("<str:Id>/adminpage3", views.adminpage3, name="adminpage3"),

    path("boothcheck", views.boothcheck, name="boothcheck"),
    path("<str:Id>/access", views.access, name="access"),
    path("<str:Id>/waiting", views.waiting, name="waiting"),
    path("<str:Id>/voterpage3", views.voterpage3, name="voterpage3"),

    path("<str:Id>/calculate", views.calculate, name="calculate"),
    path("<str:Id>/result", views.view_result, name="view_result"),

    path("history", views.history, name="history"),
    path("details", views.view_details, name="view_details"),
    path("EditPassword", views.edit_password, name="edit_password"),

    path("feedback", views.feedback, name="feedback")

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
