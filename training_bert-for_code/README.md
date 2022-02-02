# ACoRA training BERT4Code

NOTE: This is an optional activity (read it before you run any of the scripts in this folder)

BERT4Code is a baseline model trained on a whole codebase that is further used as a basis for other models and finding similarities between the lines.

The BERT4Code model that you can find in this demo was trained on:

```
https://www.bosch-ebike.com/se/connect/licences/
chromium-75.0.3770.100-patched
```

So there is probably no need to train it once again (also it is computational-consuming to do so, especially without GPU). But, if you feel it is needed, you should:

1. Copy your code into the "codebase" subfolder in this folder
2. Run the scripts (inside the container - see the main README file) in the order that is indiciated in their names - 01-.., 02-..., ...