# PokeGAN
In this repository, I show my attempt to generate new *Pokémons* using a [Deep Convolutional Generative Adversarial Network](https://arxiv.org/abs/1511.06434). 
As you can see, the little monsters are satisfying !

![MyPokemons](https://github.com/dechantoine/PokeGAN/blob/master/generated_pkmn.png)

But, how did I get those ? Let's see how to manage a DCGAN in few steps !


### 1 - Data

I have downloaded all Pokémons sprites on [veekun.com](https://veekun.com/dex/downloads): the five generations are available, from Green, Red, Blue, and Yellow to Black and White.
The icons have changed throughout the generations, in size and in style :

![AllGens](https://github.com/dechantoine/PokeGAN/blob/master/allgen.PNG)

First and second gen Bulbasaur looks ugly now... Because the actual style for Pokémons have been adopted in the third gen, I dropped icons from the oldest gen. I chose to work with 64x64 pictures because it suits the convolution and deconvolution process very well. So, I cropped icons with size greater thant 64x64 if possible (if the Pokemon on the icon can fit in a 64x64 square). Finally, I changed the white background in a black background and worked with grayscaled images.

![Process](https://github.com/dechantoine/PokeGAN/blob/master/process.PNG)

With the grayscaled images, I did a more advanced selection aiming to kept only Pokemons with eyes (and by extension the head) in the top left or mid left of the picture. So I reversed some pictures and dropped others. I ended up with around 500 icons.

![Second criterion](https://github.com/dechantoine/PokeGAN/blob/master/second_criterion.PNG)


### 2 - Design a DCGAN

To chose the architecture of my DCGAN, I followed the core proposed in the paper [*Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*](https://arxiv.org/abs/1511.06434) and some advices found and gathered by [Soumith Chintala *et al*](https://github.com/soumith/ganhacks).

![Generator](https://github.com/dechantoine/PokeGAN/blob/master/generator.jpg)

![Discriminator](https://github.com/dechantoine/PokeGAN/blob/master/discriminator.jpg)

### 3 - Training

### 4 - Filter

### 5 - Colorization

## Acknowledgments
