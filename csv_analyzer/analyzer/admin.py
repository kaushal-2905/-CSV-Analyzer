from django.contrib import admin

# Register your models here.
from .models import UploadedCSV

class UploadedCSVAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'phone', 'uploaded_at')
    search_fields = ('name', 'email')
    list_filter = ('uploaded_at',)

admin.site.register(UploadedCSV, UploadedCSVAdmin)