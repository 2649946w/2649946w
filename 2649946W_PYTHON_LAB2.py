#!/usr/bin/env python
# coding: utf-8

# 2649946W https://github.com/2649946w/2649946w.git
# 
# #Week 1: Getting Started with Anaconda, Jupyter, Notebook and Python
# Why I chose this course?
# I am passionate in regards to the development of Artificial Intelligence for a positive and ethical cause. I want to enter the Artificial Intelligence Research field.
# I believe that AI can be implemented in an effective and ethical manner in order to assist positive causes
# 
# Prior Experience?
# I have been coding Python since I was 11 and have worked on a variety of programmes from automated bot systems to basic text based video games.
# 
# What do I expect to learn?
# - New coding skills related to Artifical Intelligence
# - Discussions and theory related to the ethics of AI implementation
# - Artifical Intelligence history
# - How to work on Artificial Intelligence in a practical manner
# 
# 
# 

# In[ ]:


print ("Hello World!")


# In[ ]:


message = ("Hello Dmis")
print (message)


# In[ ]:


print (message + message)


# In[ ]:


print (message*3)


# In[ ]:


print (message[4])


# In[ ]:


#changing the number changes the letter printed corrosponding to their position in the text


# In[ ]:


from IPython.display import*
YouTubeVideo("ADnAzArC1uk")


# In[ ]:


import webbrowser
import requests
#importing the necessary libraries

print("lets find an old site, shall we?") #This prints the User Interface
site = input("Type a Website URL:") #This defines the variable 'site' and furthers UI
era = input("Type Year, Month, and Date., e.g. 20160713")#This defines the variable era and furthers UI
url = "http://archive.org/wayback/available?url=%s&timestamp=%s" % (site, era) #URL variable defined and proveds the necessary link
response = requests.get(url)#allows the programme to load the url 
data = response.json()#i do not know


try:
    old_site = data["archived_snapshots"]["closest"]["url"]#this loads the necessary page
    print("Found this copy:", old_site)#printing text for the user and uses the 'old_site' variable
    print("it should appear in your broswer now.")#prints the UI
    webbrowser.open(old_site)#Opens the web browser
except:
    print("Sorry, could not find the site")#If the page does not exsist the programme will print this


# In[ ]:


#There are 6 Variables
#2 libraries were imported


# In[ ]:


from IPython.display import Image


# In[ ]:


Image ("picture1.jpg")


# In[ ]:


from IPython.display import Audio
Audio("audio1.mid")


# In[ ]:


from IPython.display import Audio
Audio("GoldbergVariations_MehmetOkonsar-1of3_Var1to10.ogg")


# In[ ]:




