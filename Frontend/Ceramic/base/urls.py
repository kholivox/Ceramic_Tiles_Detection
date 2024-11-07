# base/urls.py
from django.urls import path
from .views import index, predictImage, HomePage, SignupPage, LoginPage, LogoutPage
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', HomePage, name='home'),
    path('index/', index, name='index'),
    path('signup/', SignupPage, name='signup'),
    path('login/', LoginPage, name='login'),
    path('logout/', LogoutPage, name='logout'),
    path('predictImage/', predictImage, name='predictImage'),
    # other paths...
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)