from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
urlpatterns = [
    path('',views.index,name='index'),
    path('_',views.index,name='index'),
    path('home',views.home,name='home'),
    path('response',views.response,name='response'),
    path('abcd',views.abcd),
    path('contact',views.contact),
    path('aboutpyfit',views.aboutpyfit)
    
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)