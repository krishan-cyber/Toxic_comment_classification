from django.shortcuts import render,redirect
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import predict,predict_full
from django.views.decorators.cache import never_cache
from django.utils.decorators import method_decorator
import re

category={0:"Hate Speech",1:"Offensive Language",2:"Normal"}

def clean_text(text):
    return re.sub(r'[^a-zA-Z]+', ' ', text.lower())



@api_view(['POST'])
def detect_hate_speech(request):
    text = request.data.get('text', '')
    text=clean_text(text)
    if text:
        result = predict_full(text)
        result2=predict(text)
        return Response({"prediction":result2,
            "class_name":category[result2],
            "probabilities": {
    "Hate Speech": result[0],
    "Offensive Language": result[1],
    "Normal": result[2],}
  })
    return Response({'error': 'No text provided'}, status=400)

@never_cache
def home(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        text=clean_text(text)
        if text.strip():
            prediction = predict(text)
            request.session['result'] = category[prediction]
        return redirect('home')  

    result = request.session.pop('result', None)  
    return render(request, 'home.html', {'result': result})