from rest_framework import serializers
from api.models import analytics


class anSerializer(serializers.ModelSerializer):

    class Meta:
        model = analytics
        fields = '__all__'