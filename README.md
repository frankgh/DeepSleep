###Git global setup

```
git config --global user.name "Your Name"
git config --global user.email "user@wpi.edu"
```

### Create a new repository

```
git clone http://solar-10.wpi.edu/ClinicalSleep/DeepSleep.git
cd DeepSleep
touch README.md
git add README.md
git commit -m "add README"
git push -u origin master
```

### Existing folder or Git repository

```
cd existing_folder
git init
git remote add origin http://solar-10.wpi.edu/ClinicalSleep/DeepSleep.git
git add .
git commit
git push -u origin master
```