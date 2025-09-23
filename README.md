<h2 align="center">KittyDiffusion😻</h2>
<p align="center">It's so easy I can even explain it to my cat.(oh, and it generates pretty cats)</p><br>
<p align="center">Results.</p><br>

<p align="center"><img src="https://i.ibb.co/sk89pMS/492914380-c01b07f7-863e-4da2-8662-0d5ffe8120c4.jpg" style="border-radius:20px" width=60%></p>

<h1 align="center">What is Stable Diffusion?</h1>
<p align="center">Stable Diffusion took the world of AI by storm.<br> It is an image generation model, which takes a text prompt (however, for simplicity, this one doesn't)<br>
and generates a high quality image matching the prompt.<br>It sounds like something complex (and at such scale it is), but you can actually implement your own mini stable diffusion<br>
in a couple of evenings! Let's see how it works.</p>


<h1 align="center">How does Stable Diffusion work?</h1>
<p align="center">Roughly speaking, Stable Diffusion takes an image from the data (the look-a-like images we want to see our model produce)<br>
It adds noise to the image and then trains to remove the noise from it.<br>
Yep, we destroy the image to train our model to reconstruct it, so that when we give our model some noise it will reconstruct it into some image.<br>
Let's see it in details.
<br>Stable Diffusion algorithm consists of 2 stages.</p>
<h3 align="center">Forward</h3>
<p align="center">On forward stage we add noise to the image. However, we need to train our network, not confuse it.<br>
We add noise of different strength to the images. For this purpose we have a noise scheduler, which controls how much noise to add and how much of an image to keep.<br>
Explore `processes.py` to see how it works under the hood (This is a high-level overview, so for great details I strongly recommend to watch <a href="https://www.youtube.com/watch?v=HoKDTa5jHvg">this video</a><br>
Long story short, we randomly select a timestep t, which defines how strong the noise level is and add noise of this strength to out image.<br>
Note: sample function runs the entire diffusion loop to generate an image.
</p>

<h3 align="center">Backward</h3>
<p align="center">On backward process we take a destroyed image and a timestep t and pass it to the model to denoise the image (we pass t to add context. without it, we confuse the model)<br>
Model predicts the noise (it's an image of the same size as input, so that we can subtract the noise - this is what we do).<br>
We gradually subtract the noise from the image (doing it in 1 step produces low-quality results)<br>Sounds like a charm, but how do we calculate the loss and what model do we use?
You're goddamn right to question this.<br>
As for loss we use a plain MSE loss, which compares predicted noise (noise from the model) with our added noise (add_noise function in processes.py returns both noisy image and the noise)<br>
U-Net serves as a model to predict the noise. It is the model from 2015, which serves segmentation task (or how's it called).<br>
If you're familiar with AE, this one reminds it a lot. On top of AE-like architecture it uses residual skip-connections and self-attention (the original model didn't, but in DDPM we do)</p>
<br>
No one is probably reading it anyway + ngl I'm a lazy ass to explain everything here, so if I ever explain DDPM (Denoising Diffusion Probabilistic Model), I'll do it in video format.<br>
There's a lot to explore: denoising latents (adding VAE to the pipeline), adding guidance, adding CLIP text embeddings to follow the prompt.<br>
This implementation is as simple as possible.<br>
I didn't add all the complex stuff on purpose (+ ngl, I'm a lazy ass)<br>

---

<p align="center">Developed by Cheslaff with love❤️.</p>
