
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Aminekabene/AI_Mask_Classifier">
    <img src="./demo/5985970.png" width="125">
  </a>

  <h3 align="center">AI Face Mask Classifier</h3>

  <p align="center">
    A CNN model that can tell which type of mask you are wearing ;)
    <br />
    <a href="https://github.com/Aminekabene/AI_Mask_Classifier">View Demo</a>
    ·
    <a href="https://github.com/Aminekabene/AI_Mask_Classifier/issues">Report Bug</a>
    ·
    <a href="https://github.com/Aminekabene/AI_Mask_Classifier/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="###The Architecture">The Architecture</a></li>
        <li><a href="### Let's Talk Numbers"> Performance</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<div align="center">
  <a href="https://github.com/Aminekabene/AI_Mask_Classifier">
      <img src="./demo/nomask.png" width="200">
  </a>
  <a href="https://github.com/Aminekabene/AI_Mask_Classifier">
      <img src="./demo/cloth.png" width="200">
  </a>
  <br />
  <a href="https://github.com/Aminekabene/AI_Mask_Classifier">
      <img src="./demo/N95.png" width="200">
  </a>
  <a href="https://github.com/Aminekabene/AI_Mask_Classifier">
      <img src="./demo/surgical.png" width="200">
  </a>
 </div>

As you can see the Classifier can distinguish between four different classes: `No Mask, Surgical Maks, Cloth Masks and N95 Masks`.

### The Architecture
The classifier was implemented using the AlexNet CNN architecture:
* 5 Convolution Layers
  * Batch Normalisation
  * Relu Activation
  * Max Pooling
* 3 Fully Connected Layers
  * Drop out rate of 0.5
  * Relu Activation

### Let's Talk Numbers

The model was trained on 50 epoch achieving **76% Accuracy**! It was trained on a custom dataset compromising of 1200 training images and 400 test images. A variety of random transforms were applied to the dataset in an effort to enhanced the performance of the model.

I know I know you want to see the accuracy, recall, f1-score, class performance, etc. Don't worry, Here it is :)

                   precision    recall  f1-score   support

           CLoth      0.53      0.89      0.66       900
           N95        0.79      0.67      0.72      2700
           No Mask    0.82      0.85      0.84      1350
           Surgical   0.87      0.77      0.82      2250

    accuracy                              0.76      7200
    macro avg         0.75      0.80      0.76      7200
    weighted avg      0.79      0.76      0.77      7200



<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![pytorch][pytorch]][pytorch-url]
* [![OpenCV][OpenCV]][OpenCV-url]
* [![Python][Python]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact Me

Amine Kabene

* Email:  kabenea99@gmail.com
* LinkedIn: [Amine Kabene](https://www.linkedin.com/in/amine-kabene/)
* Discord: **dzskillz#2196** 

Project Link: [https://github.com/Aminekabene/AI_Mask_Classifier](https://github.com/Aminekabene/AI_Mask_Classifier)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Thank you for checking out my project!



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/amine-kabene/
[pytorch]: https://img.shields.io/badge/pytorch-000000?style=for-the-badge&logo=pytorch&logoColor=red
[pytorch-url]: https://pytorch.org/
[OpenCV]: https://img.shields.io/badge/OpenCV-000000?style=for-the-badge&logo=opencv&logoColor=green
[OpenCV-url]: https://opencv.org/
[Python]: https://img.shields.io/badge/Python-000000?style=for-the-badge&logo=Python&logoColor=yellow
[Python-url]: https://www.python.org/
