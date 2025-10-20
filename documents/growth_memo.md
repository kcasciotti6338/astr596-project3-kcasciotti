# Growth Memo
## What you built and why

I actually started this project by implementing the MCRT exactly how the source code was structured, using all the given loops, classes, and functions, just to make sure I understood how everything worked. It ran correctly, but it was painfully slow. Even small runs with a $16^3$ grid and only a few thousand packets were taking minutes to finish. Since I already knew I wanted to focus on optimization for this project, I decided to take the largest slowdown factor (photon propagation) and figure out how to implement jit there. 

Once I confirmed the physics worked, I gutted the original transport and photon modules and rewrote them to be jit-friendly. That meant bypassing the photon class entirely and replacing all the object-based loops with NumPy arrays and simple functions that could be compiled with njit. I used prange for parallel loops, and added fastmath to help Numba vectorize the math operations. After those changes, it worked significantly faster. However, I then had to change how I could track the results, since my original mcrt function updated the tracker class for each photon. 

I did save most of this old code in an old_code.py file, since I wasn't sure if I would need it again, or if you wanted to see it for grading. 

## Challenges you faced (everyone has them!)

I had such a hard time figuring out how to calculate the distance to the next cell. I didn't expect that to be so challenging, but I think after that first class, Caden and I proceeded to spend 2 hrs trying to conceptualize it on the whiteboard in our office. And that was just pseudo-code, so it took me a hot minute to get that working in code as well. 

I know no one will agree with me on this, but I found the starter code to be more challenging than starting from scratch. Starting from scratch lets me understand the abstract, strategize, and build my own structures before putting in detail. Like being told to draw something and being given a blank sheet of paper to sketch on. On the flip side, having starter code is like being given a connect-the-dots and having a vague sense of what the final picture should be, but not being able to see the picture on top of the dots. It's not too difficult to start with 1 and work your way through, but then you have scribbles everywhere from when you started going the wrong way and had to turn around. I hope that made sense haha. I know it's a different skill, though, so I guess I just need to find my own strategy to work off of starter code, but projects 1, 3 did not go as smoothly for me as project 2. 

## How you used and verified AI assistance

For the majority of the project, I only used AI to bounce ideas off of, if my classmates weren't around. When I started implementing jit, though, I worked a lot more with it. I uploaded the course website and a bunch of numpy and numba documentation to notebook lm, which made it so much easier to pull up just the relevant info and get some feedback, without it spitting generated code at me. 

By the time I had the simulation running smoothly and fully optimized, I’d already spent wayyyy more time on this project than I planned. And to be honest, writing plotting functions is my least favorite part of these assignments, and it didn’t feel like a good use of time after all the physics and performance work. So, for the plotting functions, I wrote some detailed pseudo-code, describing exactly how I wanted each plot to look and what data to pass in. I then passed that pseudo-code to ChatGPT to turn it into actual, functional code. It definitely did some weird things at first, and didn't follow all of my directions, but after a while of altering and working on it, I was able to get a functional mcrt_viz module. 

## What excited or surprised you

After the last project, I knew I wanted to use jit for my extension. I was so tired of waiting for an hour for my showcase to finish. At first glance, it looked super simple: just add @jit before each function. However, the reality of implementing it into a pre-existing code built for organization and conceptualization is a lot rougher. I was surprised by how much I needed to change in my code. Jit couldn't even touch a class without getting mad. 

## What you’d explore next

Next, I'm not sure. I think jit was easier for me than the vectorization in project 2, but I'm not sure if that's just from the equations or from the method. We'll see which looks more doable for the next project. As much as I'd like to say I wish I started earlier or dedicated more time to this project, I don't think that's possible. I spend so so so much time on this.