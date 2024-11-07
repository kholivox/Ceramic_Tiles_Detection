from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.conf import settings
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import os

# Define the path to the MobileNetV2 model
MODEL_PATH = os.path.join(settings.BASE_DIR, 'mobilenetv2_model.h5')

# Load the model once at the start
model = load_model(MODEL_PATH)

def HomePage(request):
    return render(request, 'home.html')

def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        if pass1 != pass2:
            return HttpResponse("Your password and confirm password are not same!!")
        else:
            my_user = User.objects.create_user(uname, email, pass1)
            my_user.save()
            return redirect('login')

    return render(request, 'signup.html')

def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        pass1 = request.POST.get('pass')
        user = authenticate(request, username=username, password=pass1)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            return HttpResponse("Username or Password is incorrect!!!")

    return render(request, 'login.html')

def LogoutPage(request):
    logout(request)
    return redirect('home')

@login_required(login_url='login')
def index(request):
    return render(request, 'index.html')

@login_required(login_url='login')
def predictImage(request):
    if request.method == 'POST' and request.FILES['filePath']:
        # Retrieve the file from the request
        fileObj = request.FILES['filePath']
        
        # Save the file to the server using FileSystemStorage
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name, fileObj)
        filePathName = fs.url(filePathName)
        
        # Construct the file path
        testimage = '.' + filePathName
        
        # Load and preprocess the image (resize to 224x224)
        Test_image = load_img(testimage, target_size=(224, 224))
        img = np.expand_dims(np.array(Test_image), axis=0) / 255.0  # Normalize the image

        # Make a prediction
        prediction = model.predict(img)
        pred = np.argmax(prediction)

        # Map the predicted class to the corresponding class name
        class_names = {0: 'crack', 1: 'spots'}  # Adjust class names based on your model
        res = class_names.get(pred, 'Unknown')
        
        # Plot the image with the prediction result
        plt.imshow(Test_image)
        plt.title(res)
        plt.axis('off')
        
        # Save the plot to a file
        plot_path = os.path.join(settings.BASE_DIR, 'static', 'plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Pass the file path and prediction result to the template
        context = {
            'filePathName': filePathName,
            'predictedLabel': res,
            'plotPath': 'static/plot.png'
        }
        
        # Render the result in the template
        return render(request, 'index.html', context)
    else:
        return render(request, 'index.html')
