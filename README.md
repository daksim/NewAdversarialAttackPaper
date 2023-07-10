# Latest Adversarial Attack Papers
**update at 2023-07-10 10:31:19**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. When and How to Fool Explainable Models (and Humans) with Adversarial Examples**

cs.LG

Updated version. 43 pages, 9 figures, 4 tables

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2107.01943v2) [paper-pdf](http://arxiv.org/pdf/2107.01943v2)

**Authors**: Jon Vadillo, Roberto Santana, Jose A. Lozano

**Abstract**: Reliable deployment of machine learning models such as neural networks continues to be challenging due to several limitations. Some of the main shortcomings are the lack of interpretability and the lack of robustness against adversarial examples or out-of-distribution inputs. In this exploratory review, we explore the possibilities and limits of adversarial attacks for explainable machine learning models. First, we extend the notion of adversarial examples to fit in explainable machine learning scenarios, in which the inputs, the output classifications and the explanations of the model's decisions are assessed by humans. Next, we propose a comprehensive framework to study whether (and how) adversarial examples can be generated for explainable models under human assessment, introducing and illustrating novel attack paradigms. In particular, our framework considers a wide range of relevant yet often ignored factors such as the type of problem, the user expertise or the objective of the explanations, in order to identify the attack strategies that should be adopted in each scenario to successfully deceive the model (and the human). The intention of these contributions is to serve as a basis for a more rigorous and realistic study of adversarial examples in the field of explainable machine learning.



## **2. Enhancing Adversarial Training via Reweighting Optimization Trajectory**

cs.LG

Accepted by ECML 2023

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2306.14275v3) [paper-pdf](http://arxiv.org/pdf/2306.14275v3)

**Authors**: Tianjin Huang, Shiwei Liu, Tianlong Chen, Meng Fang, Li Shen, Vlaod Menkovski, Lu Yin, Yulong Pei, Mykola Pechenizkiy

**Abstract**: Despite the fact that adversarial training has become the de facto method for improving the robustness of deep neural networks, it is well-known that vanilla adversarial training suffers from daunting robust overfitting, resulting in unsatisfactory robust generalization. A number of approaches have been proposed to address these drawbacks such as extra regularization, adversarial weights perturbation, and training with more data over the last few years. However, the robust generalization improvement is yet far from satisfactory. In this paper, we approach this challenge with a brand new perspective -- refining historical optimization trajectories. We propose a new method named \textbf{Weighted Optimization Trajectories (WOT)} that leverages the optimization trajectories of adversarial training in time. We have conducted extensive experiments to demonstrate the effectiveness of WOT under various state-of-the-art adversarial attacks. Our results show that WOT integrates seamlessly with the existing adversarial training methods and consistently overcomes the robust overfitting issue, resulting in better adversarial robustness. For example, WOT boosts the robust accuracy of AT-PGD under AA-$L_{\infty}$ attack by 1.53\% $\sim$ 6.11\% and meanwhile increases the clean accuracy by 0.55\%$\sim$5.47\% across SVHN, CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.



## **3. Evaluating Similitude and Robustness of Deep Image Denoising Models via Adversarial Attack**

cs.CV

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2306.16050v2) [paper-pdf](http://arxiv.org/pdf/2306.16050v2)

**Authors**: Jie Ning, Jiebao Sun, Yao Li, Zhichang Guo, Wangmeng Zuo

**Abstract**: Deep neural networks (DNNs) have shown superior performance comparing to traditional image denoising algorithms. However, DNNs are inevitably vulnerable while facing adversarial attacks. In this paper, we propose an adversarial attack method named denoising-PGD which can successfully attack all the current deep denoising models while keep the noise distribution almost unchanged. We surprisingly find that the current mainstream non-blind denoising models (DnCNN, FFDNet, ECNDNet, BRDNet), blind denoising models (DnCNN-B, Noise2Noise, RDDCNN-B, FAN), plug-and-play (DPIR, CurvPnP) and unfolding denoising models (DeamNet) almost share the same adversarial sample set on both grayscale and color images, respectively. Shared adversarial sample set indicates that all these models are similar in term of local behaviors at the neighborhood of all the test samples. Thus, we further propose an indicator to measure the local similarity of models, called robustness similitude. Non-blind denoising models are found to have high robustness similitude across each other, while hybrid-driven models are also found to have high robustness similitude with pure data-driven non-blind denoising models. According to our robustness assessment, data-driven non-blind denoising models are the most robust. We use adversarial training to complement the vulnerability to adversarial attacks. Moreover, the model-driven image denoising BM3D shows resistance on adversarial attacks.



## **4. A Vulnerability of Attribution Methods Using Pre-Softmax Scores**

cs.LG

7 pages, 5 figures,

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03305v1) [paper-pdf](http://arxiv.org/pdf/2307.03305v1)

**Authors**: Miguel Lerma, Mirtha Lucas

**Abstract**: We discuss a vulnerability involving a category of attribution methods used to provide explanations for the outputs of convolutional neural networks working as classifiers. It is known that this type of networks are vulnerable to adversarial attacks, in which imperceptible perturbations of the input may alter the outputs of the model. In contrast, here we focus on effects that small modifications in the model may cause on the attribution method without altering the model outputs.



## **5. Quantum Solutions to the Privacy vs. Utility Tradeoff**

quant-ph

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03118v1) [paper-pdf](http://arxiv.org/pdf/2307.03118v1)

**Authors**: Sagnik Chatterjee, Vyacheslav Kungurtsev

**Abstract**: In this work, we propose a novel architecture (and several variants thereof) based on quantum cryptographic primitives with provable privacy and security guarantees regarding membership inference attacks on generative models. Our architecture can be used on top of any existing classical or quantum generative models. We argue that the use of quantum gates associated with unitary operators provides inherent advantages compared to standard Differential Privacy based techniques for establishing guaranteed security from all polynomial-time adversaries.



## **6. On Distribution-Preserving Mitigation Strategies for Communication under Cognitive Adversaries**

cs.IT

Presented at IEEE ISIT 2023

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03105v1) [paper-pdf](http://arxiv.org/pdf/2307.03105v1)

**Authors**: Soumita Hazra, J. Harshan

**Abstract**: In wireless security, cognitive adversaries are known to inject jamming energy on the victim's frequency band and monitor the same band for countermeasures thereby trapping the victim. Under the class of cognitive adversaries, we propose a new threat model wherein the adversary, upon executing the jamming attack, measures the long-term statistic of Kullback-Leibler Divergence (KLD) between its observations over each of the network frequencies before and after the jamming attack. To mitigate this adversary, we propose a new cooperative strategy wherein the victim takes the assistance for a helper node in the network to reliably communicate its message to the destination. The underlying idea is to appropriately split their energy and time resources such that their messages are reliably communicated without disturbing the statistical distribution of the samples in the network. We present rigorous analyses on the reliability and the covertness metrics at the destination and the adversary, respectively, and then synthesize tractable algorithms to obtain near-optimal division of resources between the victim and the helper. Finally, we show that the obtained near-optimal division of energy facilitates in deceiving the adversary with a KLD estimator.



## **7. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

cs.CV

23 pages, 17 figures, 1 table

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02881v1) [paper-pdf](http://arxiv.org/pdf/2307.02881v1)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating probability density functions for images that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space - not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, although images may lie on such lower-dimensional manifolds, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. In pursuing this goal, we consider generative models that are popular in AI and computer vision community. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: it should be possible to sample from this distribution according to the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute the probability of the sample, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show that such probabilistic descriptions can be used to construct defences against adversarial attacks. In addition to describing the manifold in terms of density, we also consider how semantic interpretations can be used to describe points on the manifold. To this end, we consider an emergent language framework which makes use of variational encoders to produce a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described in terms of evolving semantic descriptions.



## **8. NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic**

cs.CL

Published as a conference paper at ACL 2023

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02849v1) [paper-pdf](http://arxiv.org/pdf/2307.02849v1)

**Authors**: Zi'ou Zheng, Xiaodan Zhu

**Abstract**: Reasoning has been a central topic in artificial intelligence from the beginning. The recent progress made on distributed representation and neural networks continues to improve the state-of-the-art performance of natural language inference. However, it remains an open question whether the models perform real reasoning to reach their conclusions or rely on spurious correlations. Adversarial attacks have proven to be an important tool to help evaluate the Achilles' heel of the victim models. In this study, we explore the fundamental problem of developing attack models based on logic formalism. We propose NatLogAttack to perform systematic attacks centring around natural logic, a classical logic formalism that is traceable back to Aristotle's syllogism and has been closely developed for natural language inference. The proposed framework renders both label-preserving and label-flipping attacks. We show that compared to the existing attack models, NatLogAttack generates better adversarial examples with fewer visits to the victim models. The victim models are found to be more vulnerable under the label-flipping setting. NatLogAttack provides a tool to probe the existing and future NLI models' capacity from a key viewpoint and we hope more logic-based attacks will be further explored for understanding the desired property of reasoning.



## **9. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

cs.CV

10 pages, 6 figures, 7 tables. arXiv admin note: substantial text  overlap with arXiv:2204.02887

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02828v1) [paper-pdf](http://arxiv.org/pdf/2307.02828v1)

**Authors**: Xu Han, Anmin Liu, Chenxuan Yao, Yanbo Fan, Kun He

**Abstract**: Deep neural networks are known to be vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to the benign input. After achieving nearly 100% attack success rates in white-box setting, more focus is shifted to black-box attacks, of which the transferability of adversarial examples has gained significant attention. In either case, the common gradient-based methods generally use the sign function to generate perturbations on the gradient update, that offers a roughly correct direction and has gained great success. But little work pays attention to its possible limitation. In this work, we observe that the deviation between the original gradient and the generated noise may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability. To this end, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM). Specifically, we use data rescaling to substitute the sign function without extra computational cost. We further propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method could be used in any gradient-based attacks and is extensible to be integrated with various input transformation or ensemble methods to further improve the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our method could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.



## **10. A Testbed To Study Adversarial Cyber-Attack Strategies in Enterprise Networks**

cs.CR

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02794v1) [paper-pdf](http://arxiv.org/pdf/2307.02794v1)

**Authors**: Ayush Kumar, David K. Yau

**Abstract**: In this work, we propose a testbed environment to capture the attack strategies of an adversary carrying out a cyber-attack on an enterprise network. The testbed contains nodes with known security vulnerabilities which can be exploited by hackers. Participants can be invited to play the role of a hacker (e.g., black-hat, hacktivist) and attack the testbed. The testbed is designed such that there are multiple attack pathways available to hackers. We describe the working of the testbed components and discuss its implementation on a VMware ESXi server. Finally, we subject our testbed implementation to a few well-known cyber-attack strategies, collect data during the process and present our analysis of the data.



## **11. Chaos Theory and Adversarial Robustness**

cs.LG

14 pages, 6 figures

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2210.13235v2) [paper-pdf](http://arxiv.org/pdf/2210.13235v2)

**Authors**: Jonathan S. Kent

**Abstract**: Neural networks, being susceptible to adversarial attacks, should face a strict level of scrutiny before being deployed in critical or adversarial applications. This paper uses ideas from Chaos Theory to explain, analyze, and quantify the degree to which neural networks are susceptible to or robust against adversarial attacks. To this end, we present a new metric, the "susceptibility ratio," given by $\hat \Psi(h, \theta)$, which captures how greatly a model's output will be changed by perturbations to a given input.   Our results show that susceptibility to attack grows significantly with the depth of the model, which has safety implications for the design of neural networks for production environments. We provide experimental evidence of the relationship between $\hat \Psi$ and the post-attack accuracy of classification models, as well as a discussion of its application to tasks lacking hard decision boundaries. We also demonstrate how to quickly and easily approximate the certified robustness radii for extremely large models, which until now has been computationally infeasible to calculate directly.



## **12. GIT: Detecting Uncertainty, Out-Of-Distribution and Adversarial Samples using Gradients and Invariance Transformations**

cs.LG

Accepted at IJCNN 2023

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02672v1) [paper-pdf](http://arxiv.org/pdf/2307.02672v1)

**Authors**: Julia Lust, Alexandru P. Condurache

**Abstract**: Deep neural networks tend to make overconfident predictions and often require additional detectors for misclassifications, particularly for safety-critical applications. Existing detection methods usually only focus on adversarial attacks or out-of-distribution samples as reasons for false predictions. However, generalization errors occur due to diverse reasons often related to poorly learning relevant invariances. We therefore propose GIT, a holistic approach for the detection of generalization errors that combines the usage of gradient information and invariance transformations. The invariance transformations are designed to shift misclassified samples back into the generalization area of the neural network, while the gradient information measures the contradiction between the initial prediction and the corresponding inherent computations of the neural network using the transformed sample. Our experiments demonstrate the superior performance of GIT compared to the state-of-the-art on a variety of network architectures, problem setups and perturbation types.



## **13. Jailbroken: How Does LLM Safety Training Fail?**

cs.LG

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02483v1) [paper-pdf](http://arxiv.org/pdf/2307.02483v1)

**Authors**: Alexander Wei, Nika Haghtalab, Jacob Steinhardt

**Abstract**: Large language models trained for safety and harmlessness remain susceptible to adversarial misuse, as evidenced by the prevalence of "jailbreak" attacks on early releases of ChatGPT that elicit undesired behavior. Going beyond recognition of the issue, we investigate why such attacks succeed and how they can be created. We hypothesize two failure modes of safety training: competing objectives and mismatched generalization. Competing objectives arise when a model's capabilities and safety goals conflict, while mismatched generalization occurs when safety training fails to generalize to a domain for which capabilities exist. We use these failure modes to guide jailbreak design and then evaluate state-of-the-art models, including OpenAI's GPT-4 and Anthropic's Claude v1.3, against both existing and newly designed attacks. We find that vulnerabilities persist despite the extensive red-teaming and safety-training efforts behind these models. Notably, new attacks utilizing our failure modes succeed on every prompt in a collection of unsafe requests from the models' red-teaming evaluation sets and outperform existing ad hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity -- that safety mechanisms should be as sophisticated as the underlying model -- and argues against the idea that scaling alone can resolve these safety failure modes.



## **14. Defense against Adversarial Cloud Attack on Remote Sensing Salient Object Detection**

cs.CV

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2306.17431v2) [paper-pdf](http://arxiv.org/pdf/2306.17431v2)

**Authors**: Huiming Sun, Lan Fu, Jinlong Li, Qing Guo, Zibo Meng, Tianyun Zhang, Yuewei Lin, Hongkai Yu

**Abstract**: Detecting the salient objects in a remote sensing image has wide applications for the interdisciplinary research. Many existing deep learning methods have been proposed for Salient Object Detection (SOD) in remote sensing images and get remarkable results. However, the recent adversarial attack examples, generated by changing a few pixel values on the original remote sensing image, could result in a collapse for the well-trained deep learning based SOD model. Different with existing methods adding perturbation to original images, we propose to jointly tune adversarial exposure and additive perturbation for attack and constrain image close to cloudy image as Adversarial Cloud. Cloud is natural and common in remote sensing images, however, camouflaging cloud based adversarial attack and defense for remote sensing images are not well studied before. Furthermore, we design DefenseNet as a learn-able pre-processing to the adversarial cloudy images so as to preserve the performance of the deep learning based remote sensing SOD model, without tuning the already deployed deep SOD model. By considering both regular and generalized adversarial examples, the proposed DefenseNet can defend the proposed Adversarial Cloud in white-box setting and other attack methods in black-box setting. Experimental results on a synthesized benchmark from the public remote sensing SOD dataset (EORSSD) show the promising defense against adversarial cloud attacks.



## **15. On the Adversarial Robustness of Generative Autoencoders in the Latent Space**

cs.LG

18 pages, 12 figures

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02202v1) [paper-pdf](http://arxiv.org/pdf/2307.02202v1)

**Authors**: Mingfei Lu, Badong Chen

**Abstract**: The generative autoencoders, such as the variational autoencoders or the adversarial autoencoders, have achieved great success in lots of real-world applications, including image generation, and signal communication.   However, little concern has been devoted to their robustness during practical deployment.   Due to the probabilistic latent structure, variational autoencoders (VAEs) may confront problems such as a mismatch between the posterior distribution of the latent and real data manifold, or discontinuity in the posterior distribution of the latent.   This leaves a back door for malicious attackers to collapse VAEs from the latent space, especially in scenarios where the encoder and decoder are used separately, such as communication and compressed sensing.   In this work, we provide the first study on the adversarial robustness of generative autoencoders in the latent space.   Specifically, we empirically demonstrate the latent vulnerability of popular generative autoencoders through attacks in the latent space.   We also evaluate the difference between variational autoencoders and their deterministic variants and observe that the latter performs better in latent robustness.   Meanwhile, we identify a potential trade-off between the adversarial robustness and the degree of the disentanglement of the latent codes.   Additionally, we also verify the feasibility of improvement for the latent robustness of VAEs through adversarial training.   In summary, we suggest concerning the adversarial latent robustness of the generative autoencoders, analyze several robustness-relative issues, and give some insights into a series of key challenges.



## **16. Boosting Adversarial Transferability via Fusing Logits of Top-1 Decomposed Feature**

cs.CV

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2305.01361v3) [paper-pdf](http://arxiv.org/pdf/2305.01361v3)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li, Zhun Zhong

**Abstract**: Recent research has shown that Deep Neural Networks (DNNs) are highly vulnerable to adversarial samples, which are highly transferable and can be used to attack other unknown black-box models. To improve the transferability of adversarial samples, several feature-based adversarial attack methods have been proposed to disrupt neuron activation in the middle layers. However, current state-of-the-art feature-based attack methods typically require additional computation costs for estimating the importance of neurons. To address this challenge, we propose a Singular Value Decomposition (SVD)-based feature-level attack method. Our approach is inspired by the discovery that eigenvectors associated with the larger singular values decomposed from the middle layer features exhibit superior generalization and attention properties. Specifically, we conduct the attack by retaining the decomposed Top-1 singular value-associated feature for computing the output logits, which are then combined with the original logits to optimize adversarial examples. Our extensive experimental results verify the effectiveness of our proposed method, which can be easily integrated into various baselines to significantly enhance the transferability of adversarial samples for disturbing normally trained CNNs and advanced defense strategies. The source code of this study is available at https://github.com/WJJLL/SVD-SSA



## **17. Adversarial Attacks on Image Classification Models: FGSM and Patch Attacks and their Impact**

cs.CV

This is the preprint of the chapter titled "Adversarial Attacks on  Image Classification Models: FGSM and Patch Attacks and their Impact" which  will be published in the volume titled "Information Security and Privacy in  the Digital World - Some Selected Cases", edited by Jaydip Sen. The book will  be published by IntechOpen, London, UK, in 2023. This is not the final  version of the chapter

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02055v1) [paper-pdf](http://arxiv.org/pdf/2307.02055v1)

**Authors**: Jaydip Sen, Subhasis Dasgupta

**Abstract**: This chapter introduces the concept of adversarial attacks on image classification models built on convolutional neural networks (CNN). CNNs are very popular deep-learning models which are used in image classification tasks. However, very powerful and pre-trained CNN models working very accurately on image datasets for image classification tasks may perform disastrously when the networks are under adversarial attacks. In this work, two very well-known adversarial attacks are discussed and their impact on the performance of image classifiers is analyzed. These two adversarial attacks are the fast gradient sign method (FGSM) and adversarial patch attack. These attacks are launched on three powerful pre-trained image classifier architectures, ResNet-34, GoogleNet, and DenseNet-161. The classification accuracy of the models in the absence and presence of the two attacks are computed on images from the publicly accessible ImageNet dataset. The results are analyzed to evaluate the impact of the attacks on the image classification task.



## **18. Complex Graph Laplacian Regularizer for Inferencing Grid States**

eess.SP

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01906v1) [paper-pdf](http://arxiv.org/pdf/2307.01906v1)

**Authors**: Chinthaka Dinesh, Junfei Wang, Gene Cheung, Pirathayini Srikantha

**Abstract**: In order to maintain stable grid operations, system monitoring and control processes require the computation of grid states (e.g. voltage magnitude and angles) at high granularity. It is necessary to infer these grid states from measurements generated by a limited number of sensors like phasor measurement units (PMUs) that can be subjected to delays and losses due to channel artefacts, and/or adversarial attacks (e.g. denial of service, jamming, etc.). We propose a novel graph signal processing (GSP) based algorithm to interpolate states of the entire grid from observations of a small number of grid measurements. It is a two-stage process, where first an underlying Hermitian graph is learnt empirically from existing grid datasets. Then, the graph is used to interpolate missing grid signal samples in linear time. With our proposal, we can effectively reconstruct grid signals with significantly smaller number of observations when compared to existing traditional approaches (e.g. state estimation). In contrast to existing GSP approaches, we do not require knowledge of the underlying grid structure and parameters and are able to guarantee fast spectral optimization. We demonstrate the computational efficacy and accuracy of our proposal via practical studies conducted on the IEEE 118 bus system.



## **19. Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling**

cs.CV

Accepted by CVPR 2023

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01778v1) [paper-pdf](http://arxiv.org/pdf/2307.01778v1)

**Authors**: Zhanhao Hu, Wenda Chu, Xiaopei Zhu, Hui Zhang, Bo Zhang, Xiaolin Hu

**Abstract**: Recent works have proposed to craft adversarial clothes for evading person detectors, while they are either only effective at limited viewing angles or very conspicuous to humans. We aim to craft adversarial texture for clothes based on 3D modeling, an idea that has been used to craft rigid adversarial objects such as a 3D-printed turtle. Unlike rigid objects, humans and clothes are non-rigid, leading to difficulties in physical realization. In order to craft natural-looking adversarial clothes that can evade person detectors at multiple viewing angles, we propose adversarial camouflage textures (AdvCaT) that resemble one kind of the typical textures of daily clothes, camouflage textures. We leverage the Voronoi diagram and Gumbel-softmax trick to parameterize the camouflage textures and optimize the parameters via 3D modeling. Moreover, we propose an efficient augmentation pipeline on 3D meshes combining topologically plausible projection (TopoProj) and Thin Plate Spline (TPS) to narrow the gap between digital and real-world objects. We printed the developed 3D texture pieces on fabric materials and tailored them into T-shirts and trousers. Experiments show high attack success rates of these clothes against multiple detectors.



## **20. vWitness: Certifying Web Page Interactions with Computer Vision**

cs.CR

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2007.15805v2) [paper-pdf](http://arxiv.org/pdf/2007.15805v2)

**Authors**: He Shuang, Lianying Zhao, David Lie

**Abstract**: Web servers service client requests, some of which might cause the web server to perform security-sensitive operations (e.g. money transfer, voting). An attacker may thus forge or maliciously manipulate such requests by compromising a web client. Unfortunately, a web server has no way of knowing whether the client from which it receives a request has been compromised or not -- current "best practice" defenses such as user authentication or network encryption cannot aid a server as they all assume web client integrity. To address this shortcoming, we propose vWitness, which "witnesses" the interactions of a user with a web page and certifies whether they match a specification provided by the web server, enabling the web server to know that the web request is user-intended. The main challenge that vWitness overcomes is that even benign clients introduce unpredictable variations in the way they render web pages. vWitness differentiates between these benign variations and malicious manipulation using computer vision, allowing it to certify to the web server that 1) the web page user interface is properly displayed 2) observed user interactions are used to construct the web request. Our vWitness prototype achieves compatibility with modern web pages, is resilient to adversarial example attacks and is accurate and performant -- vWitness achieves 99.97% accuracy and adds 197ms of overhead to the entire interaction session in the average case.



## **21. Interpretable Computer Vision Models through Adversarial Training: Unveiling the Robustness-Interpretability Connection**

cs.CV

13 pages, 19 figures, 6 tables

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.02500v1) [paper-pdf](http://arxiv.org/pdf/2307.02500v1)

**Authors**: Delyan Boychev

**Abstract**: With the perpetual increase of complexity of the state-of-the-art deep neural networks, it becomes a more and more challenging task to maintain their interpretability. Our work aims to evaluate the effects of adversarial training utilized to produce robust models - less vulnerable to adversarial attacks. It has been shown to make computer vision models more interpretable. Interpretability is as essential as robustness when we deploy the models to the real world. To prove the correlation between these two problems, we extensively examine the models using local feature-importance methods (SHAP, Integrated Gradients) and feature visualization techniques (Representation Inversion, Class Specific Image Generation). Standard models, compared to robust are more susceptible to adversarial attacks, and their learned representations are less meaningful to humans. Conversely, these models focus on distinctive regions of the images which support their predictions. Moreover, the features learned by the robust model are closer to the real ones.



## **22. LEAT: Towards Robust Deepfake Disruption in Real-World Scenarios via Latent Ensemble Attack**

cs.CV

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01520v1) [paper-pdf](http://arxiv.org/pdf/2307.01520v1)

**Authors**: Joonkyo Shim, Hyunsoo Yoon

**Abstract**: Deepfakes, malicious visual contents created by generative models, pose an increasingly harmful threat to society. To proactively mitigate deepfake damages, recent studies have employed adversarial perturbation to disrupt deepfake model outputs. However, previous approaches primarily focus on generating distorted outputs based on only predetermined target attributes, leading to a lack of robustness in real-world scenarios where target attributes are unknown. Additionally, the transferability of perturbations between two prominent generative models, Generative Adversarial Networks (GANs) and Diffusion Models, remains unexplored. In this paper, we emphasize the importance of target attribute-transferability and model-transferability for achieving robust deepfake disruption. To address this challenge, we propose a simple yet effective disruption method called Latent Ensemble ATtack (LEAT), which attacks the independent latent encoding process. By disrupting the latent encoding process, it generates distorted output images in subsequent generation processes, regardless of the given target attributes. This target attribute-agnostic attack ensures robust disruption even when the target attributes are unknown. Additionally, we introduce a Normalized Gradient Ensemble strategy that effectively aggregates gradients for iterative gradient attacks, enabling simultaneous attacks on various types of deepfake models, involving both GAN-based and Diffusion-based models. Moreover, we demonstrate the insufficiency of evaluating disruption quality solely based on pixel-level differences. As a result, we propose an alternative protocol for comprehensively evaluating the success of defense. Extensive experiments confirm the efficacy of our method in disrupting deepfakes in real-world scenarios, reporting a higher defense success rate compared to previous methods.



## **23. SCAT: Robust Self-supervised Contrastive Learning via Adversarial Training for Text Classification**

cs.CL

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01488v1) [paper-pdf](http://arxiv.org/pdf/2307.01488v1)

**Authors**: Junjie Wu, Dit-Yan Yeung

**Abstract**: Despite their promising performance across various natural language processing (NLP) tasks, current NLP systems are vulnerable to textual adversarial attacks. To defend against these attacks, most existing methods apply adversarial training by incorporating adversarial examples. However, these methods have to rely on ground-truth labels to generate adversarial examples, rendering it impractical for large-scale model pre-training which is commonly used nowadays for NLP and many other tasks. In this paper, we propose a novel learning framework called SCAT (Self-supervised Contrastive Learning via Adversarial Training), which can learn robust representations without requiring labeled data. Specifically, SCAT modifies random augmentations of the data in a fully labelfree manner to generate adversarial examples. Adversarial training is achieved by minimizing the contrastive loss between the augmentations and their adversarial counterparts. We evaluate SCAT on two text classification datasets using two state-of-the-art attack schemes proposed recently. Our results show that SCAT can not only train robust language models from scratch, but it can also significantly improve the robustness of existing pre-trained language models. Moreover, to demonstrate its flexibility, we show that SCAT can also be combined with supervised adversarial training to further enhance model robustness.



## **24. Web3Recommend: Decentralised recommendations with trust and relevance**

cs.DC

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01411v1) [paper-pdf](http://arxiv.org/pdf/2307.01411v1)

**Authors**: Rohan Madhwal, Johan Pouwelse

**Abstract**: Web3Recommend is a decentralized Social Recommender System implementation that enables Web3 Platforms on Android to generate recommendations that balance trust and relevance. Generating recommendations in decentralized networks is a non-trivial problem because these networks lack a global perspective due to the absence of a central authority. Further, decentralized networks are prone to Sybil Attacks in which a single malicious user can generate multiple fake or Sybil identities. Web3Recommend relies on a novel graph-based content recommendation design inspired by GraphJet, a recommendation system used in Twitter enhanced with MeritRank, a decentralized reputation scheme that provides Sybil-resistance to the system. By adding MeritRank's decay parameters to the vanilla Social Recommender Systems' personalized SALSA graph algorithm, we can provide theoretical guarantees against Sybil Attacks in the generated recommendations. Similar to GraphJet, we focus on generating real-time recommendations by only acting on recent interactions in the social network, allowing us to cater temporally contextual recommendations while keeping a tight bound on the memory usage in resource-constrained devices, allowing for a seamless user experience. As a proof-of-concept, we integrate our system with MusicDAO, an open-source Web3 music-sharing platform, to generate personalized, real-time recommendations. Thus, we provide the first Sybil-resistant Social Recommender System, allowing real-time recommendations beyond classic user-based collaborative filtering. The system is also rigorously tested with extensive unit and integration tests. Further, our experiments demonstrate the trust-relevance balance of recommendations against multiple adversarial strategies in a test network generated using data from real music platforms.



## **25. Adversarial Learning in Real-World Fraud Detection: Challenges and Perspectives**

cs.LG

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2307.01390v1) [paper-pdf](http://arxiv.org/pdf/2307.01390v1)

**Authors**: Danele Lunghi, Alkis Simitsis, Olivier Caelen, Gianluca Bontempi

**Abstract**: Data economy relies on data-driven systems and complex machine learning applications are fueled by them. Unfortunately, however, machine learning models are exposed to fraudulent activities and adversarial attacks, which threaten their security and trustworthiness. In the last decade or so, the research interest on adversarial machine learning has grown significantly, revealing how learning applications could be severely impacted by effective attacks. Although early results of adversarial machine learning indicate the huge potential of the approach to specific domains such as image processing, still there is a gap in both the research literature and practice regarding how to generalize adversarial techniques in other domains and applications. Fraud detection is a critical defense mechanism for data economy, as it is for other applications as well, which poses several challenges for machine learning. In this work, we describe how attacks against fraud detection systems differ from other applications of adversarial machine learning, and propose a number of interesting directions to bridge this gap.



## **26. When Can Linear Learners be Robust to Indiscriminate Poisoning Attacks?**

cs.LG

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2307.01073v1) [paper-pdf](http://arxiv.org/pdf/2307.01073v1)

**Authors**: Fnu Suya, Xiao Zhang, Yuan Tian, David Evans

**Abstract**: We study indiscriminate poisoning for linear learners where an adversary injects a few crafted examples into the training data with the goal of forcing the induced model to incur higher test error. Inspired by the observation that linear learners on some datasets are able to resist the best known attacks even without any defenses, we further investigate whether datasets can be inherently robust to indiscriminate poisoning attacks for linear learners. For theoretical Gaussian distributions, we rigorously characterize the behavior of an optimal poisoning attack, defined as the poisoning strategy that attains the maximum risk of the induced model at a given poisoning budget. Our results prove that linear learners can indeed be robust to indiscriminate poisoning if the class-wise data distributions are well-separated with low variance and the size of the constraint set containing all permissible poisoning points is also small. These findings largely explain the drastic variation in empirical attack performance of the state-of-the-art poisoning attacks on linear learners across benchmark datasets, making an important initial step towards understanding the underlying reasons some learning tasks are vulnerable to data poisoning attacks.



## **27. Enhancing the Robustness of QMIX against State-adversarial Attacks**

cs.LG

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2307.00907v1) [paper-pdf](http://arxiv.org/pdf/2307.00907v1)

**Authors**: Weiran Guo, Guanjun Liu, Ziyuan Zhou, Ling Wang, Jiacun Wang

**Abstract**: Deep reinforcement learning (DRL) performance is generally impacted by state-adversarial attacks, a perturbation applied to an agent's observation. Most recent research has concentrated on robust single-agent reinforcement learning (SARL) algorithms against state-adversarial attacks. Still, there has yet to be much work on robust multi-agent reinforcement learning. Using QMIX, one of the popular cooperative multi-agent reinforcement algorithms, as an example, we discuss four techniques to improve the robustness of SARL algorithms and extend them to multi-agent scenarios. To increase the robustness of multi-agent reinforcement learning (MARL) algorithms, we train models using a variety of attacks in this research. We then test the models taught using the other attacks by subjecting them to the corresponding attacks throughout the training phase. In this way, we organize and summarize techniques for enhancing robustness when used with MARL.



## **28. Data Poisoning Attack Aiming the Vulnerability of Continual Learning**

cs.LG

ICIP 2023 (NeurIPS 2022 ML Safety Workshop accepted paper)

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2211.15875v2) [paper-pdf](http://arxiv.org/pdf/2211.15875v2)

**Authors**: Gyojin Han, Jaehyun Choi, Hyeong Gwon Hong, Junmo Kim

**Abstract**: Generally, regularization-based continual learning models limit access to the previous task data to imitate the real-world constraints related to memory and privacy. However, this introduces a problem in these models by not being able to track the performance on each task. In essence, current continual learning methods are susceptible to attacks on previous tasks. We demonstrate the vulnerability of regularization-based continual learning methods by presenting a simple task-specific data poisoning attack that can be used in the learning process of a new task. Training data generated by the proposed attack causes performance degradation on a specific task targeted by the attacker. We experiment with the attack on the two representative regularization-based continual learning methods, Elastic Weight Consolidation (EWC) and Synaptic Intelligence (SI), trained with variants of MNIST dataset. The experiment results justify the vulnerability proposed in this paper and demonstrate the importance of developing continual learning models that are robust to adversarial attacks.



## **29. Evaluating the Adversarial Robustness of Convolution-based Human Motion Prediction**

cs.CV

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2306.11990v2) [paper-pdf](http://arxiv.org/pdf/2306.11990v2)

**Authors**: Chengxu Duan, Zhicheng Zhang, Xiaoli Liu, Yonghao Dang, Jianqin Yin

**Abstract**: Human motion prediction has achieved a brilliant performance with the help of CNNs, which facilitates human-machine cooperation. However, currently, there is no work evaluating the potential risk in human motion prediction when facing adversarial attacks, which may cause danger in real applications. The adversarial attack will face two problems against human motion prediction: 1. For naturalness, pose data is highly related to the physical dynamics of human skeletons where Lp norm constraints cannot constrain the adversarial example well; 2. Unlike the pixel value in images, pose data is diverse at scale because of the different acquisition equipment and the data processing, which makes it hard to set fixed parameters to perform attacks. To solve the problems above, we propose a new adversarial attack method that perturbs the input human motion sequence by maximizing the prediction error with physical constraints. Specifically, we introduce a novel adaptable scheme that facilitates the attack to suit the scale of the target pose and two physical constraints to enhance the imperceptibility of the adversarial example. The evaluating experiments on three datasets show that the prediction errors of all target models are enlarged significantly, which means current convolution-based human motion prediction models can be easily disturbed under the proposed attack. The quantitative analysis shows that prior knowledge and semantic information modeling can be the key to the adversarial robustness of human motion predictors. The qualitative results indicate that the adversarial sample is hard to be noticed when compared frame by frame but is relatively easy to be detected when the sample is animated.



## **30. Sneaky Spikes: Uncovering Stealthy Backdoor Attacks in Spiking Neural Networks with Neuromorphic Data**

cs.CR

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2302.06279v2) [paper-pdf](http://arxiv.org/pdf/2302.06279v2)

**Authors**: Gorka Abad, Oguzhan Ersoy, Stjepan Picek, Aitor Urbieta

**Abstract**: Deep neural networks (DNNs) have demonstrated remarkable performance across various tasks, including image and speech recognition. However, maximizing the effectiveness of DNNs requires meticulous optimization of numerous hyperparameters and network parameters through training. Moreover, high-performance DNNs entail many parameters, which consume significant energy during training. In order to overcome these challenges, researchers have turned to spiking neural networks (SNNs), which offer enhanced energy efficiency and biologically plausible data processing capabilities, rendering them highly suitable for sensory data tasks, particularly in neuromorphic data. Despite their advantages, SNNs, like DNNs, are susceptible to various threats, including adversarial examples and backdoor attacks. Yet, the field of SNNs still needs to be explored in terms of understanding and countering these attacks.   This paper delves into backdoor attacks in SNNs using neuromorphic datasets and diverse triggers. Specifically, we explore backdoor triggers within neuromorphic data that can manipulate their position and color, providing a broader scope of possibilities than conventional triggers in domains like images. We present various attack strategies, achieving an attack success rate of up to 100\% while maintaining a negligible impact on clean accuracy. Furthermore, we assess these attacks' stealthiness, revealing that our most potent attacks possess significant stealth capabilities. Lastly, we adapt several state-of-the-art defenses from the image domain, evaluating their efficacy on neuromorphic data and uncovering instances where they fall short, leading to compromised performance.



## **31. Feature Partition Aggregation: A Fast Certified Defense Against a Union of $\ell_0$ Attacks**

cs.LG

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2302.11628v2) [paper-pdf](http://arxiv.org/pdf/2302.11628v2)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstract**: Sparse or $\ell_0$ adversarial attacks arbitrarily perturb an unknown subset of the features. $\ell_0$ robustness analysis is particularly well-suited for heterogeneous (tabular) data where features have different types or scales. State-of-the-art $\ell_0$ certified defenses are based on randomized smoothing and apply to evasion attacks only. This paper proposes feature partition aggregation (FPA) -- a certified defense against the union of $\ell_0$ evasion, backdoor, and poisoning attacks. FPA generates its stronger robustness guarantees via an ensemble whose submodels are trained on disjoint feature sets. Compared to state-of-the-art $\ell_0$ defenses, FPA is up to 3,000${\times}$ faster and provides larger median robustness guarantees (e.g., median certificates of 13 pixels over 10 for CIFAR10, 12 pixels over 10 for MNIST, 4 features over 1 for Weather, and 3 features over 1 for Ames), meaning FPA provides the additional dimensions of robustness essentially for free.



## **32. Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT)**

cs.CL

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2307.01225v1) [paper-pdf](http://arxiv.org/pdf/2307.01225v1)

**Authors**: Bushra Sabir, M. Ali Babar, Sharif Abuadbba

**Abstract**: Transformer-based text classifiers like BERT, Roberta, T5, and GPT-3 have shown impressive performance in NLP. However, their vulnerability to adversarial examples poses a security risk. Existing defense methods lack interpretability, making it hard to understand adversarial classifications and identify model vulnerabilities. To address this, we propose the Interpretability and Transparency-Driven Detection and Transformation (IT-DT) framework. It focuses on interpretability and transparency in detecting and transforming textual adversarial examples. IT-DT utilizes techniques like attention maps, integrated gradients, and model feedback for interpretability during detection. This helps identify salient features and perturbed words contributing to adversarial classifications. In the transformation phase, IT-DT uses pre-trained embeddings and model feedback to generate optimal replacements for perturbed words. By finding suitable substitutions, we aim to convert adversarial examples into non-adversarial counterparts that align with the model's intended behavior while preserving the text's meaning. Transparency is emphasized through human expert involvement. Experts review and provide feedback on detection and transformation results, enhancing decision-making, especially in complex scenarios. The framework generates insights and threat intelligence empowering analysts to identify vulnerabilities and improve model robustness. Comprehensive experiments demonstrate the effectiveness of IT-DT in detecting and transforming adversarial examples. The approach enhances interpretability, provides transparency, and enables accurate identification and successful transformation of adversarial inputs. By combining technical analysis and human expertise, IT-DT significantly improves the resilience and trustworthiness of transformer-based text classifiers against adversarial attacks.



## **33. From ChatGPT to ThreatGPT: Impact of Generative AI in Cybersecurity and Privacy**

cs.CR

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2307.00691v1) [paper-pdf](http://arxiv.org/pdf/2307.00691v1)

**Authors**: Maanak Gupta, CharanKumar Akiri, Kshitiz Aryal, Eli Parker, Lopamudra Praharaj

**Abstract**: Undoubtedly, the evolution of Generative AI (GenAI) models has been the highlight of digital transformation in the year 2022. As the different GenAI models like ChatGPT and Google Bard continue to foster their complexity and capability, it's critical to understand its consequences from a cybersecurity perspective. Several instances recently have demonstrated the use of GenAI tools in both the defensive and offensive side of cybersecurity, and focusing on the social, ethical and privacy implications this technology possesses. This research paper highlights the limitations, challenges, potential risks, and opportunities of GenAI in the domain of cybersecurity and privacy. The work presents the vulnerabilities of ChatGPT, which can be exploited by malicious users to exfiltrate malicious information bypassing the ethical constraints on the model. This paper demonstrates successful example attacks like Jailbreaks, reverse psychology, and prompt injection attacks on the ChatGPT. The paper also investigates how cyber offenders can use the GenAI tools in developing cyber attacks, and explore the scenarios where ChatGPT can be used by adversaries to create social engineering attacks, phishing attacks, automated hacking, attack payload generation, malware creation, and polymorphic malware. This paper then examines defense techniques and uses GenAI tools to improve security measures, including cyber defense automation, reporting, threat intelligence, secure code generation and detection, attack identification, developing ethical guidelines, incidence response plans, and malware detection. We will also discuss the social, legal, and ethical implications of ChatGPT. In conclusion, the paper highlights open challenges and future directions to make this GenAI secure, safe, trustworthy, and ethical as the community understands its cybersecurity impacts.



## **34. Soft Actor-Critic Algorithm with Truly-satisfied Inequality Constraint**

cs.LG

10 pages, 9 figures

**SubmitDate**: 2023-07-02    [abs](http://arxiv.org/abs/2303.04356v2) [paper-pdf](http://arxiv.org/pdf/2303.04356v2)

**Authors**: Taisuke Kobayashi

**Abstract**: Soft actor-critic (SAC) in reinforcement learning is expected to be one of the next-generation robot control schemes. Its ability to maximize policy entropy would make a robotic controller robust to noise and perturbation, which is useful for real-world robot applications. However, the priority of maximizing the policy entropy is automatically tuned in the current implementation, the rule of which can be interpreted as one for equality constraint, binding the policy entropy into its specified lower bound. The current SAC is therefore no longer maximize the policy entropy, contrary to our expectation. To resolve this issue in SAC, this paper improves its implementation with a learnable state-dependent slack variable for appropriately handling the inequality constraint to maximize the policy entropy by reformulating it as the corresponding equality constraint. The introduced slack variable is optimized by a switching-type loss function that takes into account the dual objectives of satisfying the equality constraint and checking the lower bound. In Mujoco and Pybullet simulators, the modified SAC statistically achieved the higher robustness for adversarial attacks than before while regularizing the norm of action. A real-robot variable impedance task was demonstrated for showing the applicability of the modified SAC to real-world robot control. In particular, the modified SAC maintained adaptive behaviors for physical human-robot interaction, which had no experience at all during training. https://youtu.be/EH3xVtlVaJw



## **35. X-Detect: Explainable Adversarial Patch Detection for Object Detectors in Retail**

cs.CV

**SubmitDate**: 2023-07-02    [abs](http://arxiv.org/abs/2306.08422v2) [paper-pdf](http://arxiv.org/pdf/2306.08422v2)

**Authors**: Omer Hofman, Amit Giloni, Yarin Hayun, Ikuya Morikawa, Toshiya Shimizu, Yuval Elovici, Asaf Shabtai

**Abstract**: Object detection models, which are widely used in various domains (such as retail), have been shown to be vulnerable to adversarial attacks. Existing methods for detecting adversarial attacks on object detectors have had difficulty detecting new real-life attacks. We present X-Detect, a novel adversarial patch detector that can: i) detect adversarial samples in real time, allowing the defender to take preventive action; ii) provide explanations for the alerts raised to support the defender's decision-making process, and iii) handle unfamiliar threats in the form of new attacks. Given a new scene, X-Detect uses an ensemble of explainable-by-design detectors that utilize object extraction, scene manipulation, and feature transformation techniques to determine whether an alert needs to be raised. X-Detect was evaluated in both the physical and digital space using five different attack scenarios (including adaptive attacks) and the COCO dataset and our new Superstore dataset. The physical evaluation was performed using a smart shopping cart setup in real-world settings and included 17 adversarial patch attacks recorded in 1,700 adversarial videos. The results showed that X-Detect outperforms the state-of-the-art methods in distinguishing between benign and adversarial scenes for all attack scenarios while maintaining a 0% FPR (no false alarms) and providing actionable explanations for the alerts raised. A demo is available.



## **36. Query-Efficient Decision-based Black-Box Patch Attack**

cs.CV

**SubmitDate**: 2023-07-02    [abs](http://arxiv.org/abs/2307.00477v1) [paper-pdf](http://arxiv.org/pdf/2307.00477v1)

**Authors**: Zhaoyu Chen, Bo Li, Shuang Wu, Shouhong Ding, Wenqiang Zhang

**Abstract**: Deep neural networks (DNNs) have been showed to be highly vulnerable to imperceptible adversarial perturbations. As a complementary type of adversary, patch attacks that introduce perceptible perturbations to the images have attracted the interest of researchers. Existing patch attacks rely on the architecture of the model or the probabilities of predictions and perform poorly in the decision-based setting, which can still construct a perturbation with the minimal information exposed -- the top-1 predicted label. In this work, we first explore the decision-based patch attack. To enhance the attack efficiency, we model the patches using paired key-points and use targeted images as the initialization of patches, and parameter optimizations are all performed on the integer domain. Then, we propose a differential evolutionary algorithm named DevoPatch for query-efficient decision-based patch attacks. Experiments demonstrate that DevoPatch outperforms the state-of-the-art black-box patch attacks in terms of patch area and attack success rate within a given query budget on image classification and face verification. Additionally, we conduct the vulnerability evaluation of ViT and MLP on image classification in the decision-based patch attack setting for the first time. Using DevoPatch, we can evaluate the robustness of models to black-box patch attacks. We believe this method could inspire the design and deployment of robust vision models based on various DNN architectures in the future.



## **37. Brightness-Restricted Adversarial Attack Patch**

cs.CV

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00421v1) [paper-pdf](http://arxiv.org/pdf/2307.00421v1)

**Authors**: Mingzhen Shao

**Abstract**: Adversarial attack patches have gained increasing attention due to their practical applicability in physical-world scenarios. However, the bright colors used in attack patches represent a significant drawback, as they can be easily identified by human observers. Moreover, even though these attacks have been highly successful in deceiving target networks, which specific features of the attack patch contribute to its success are still unknown. Our paper introduces a brightness-restricted patch (BrPatch) that uses optical characteristics to effectively reduce conspicuousness while preserving image independence. We also conducted an analysis of the impact of various image features (such as color, texture, noise, and size) on the effectiveness of an attack patch in physical-world deployment. Our experiments show that attack patches exhibit strong redundancy to brightness and are resistant to color transfer and noise. Based on our findings, we propose some additional methods to further reduce the conspicuousness of BrPatch. Our findings also explain the robustness of attack patches observed in physical-world scenarios.



## **38. CasTGAN: Cascaded Generative Adversarial Network for Realistic Tabular Data Synthesis**

cs.LG

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00384v1) [paper-pdf](http://arxiv.org/pdf/2307.00384v1)

**Authors**: Abdallah Alshantti, Damiano Varagnolo, Adil Rasheed, Aria Rahmati, Frank Westad

**Abstract**: Generative adversarial networks (GANs) have drawn considerable attention in recent years for their proven capability in generating synthetic data which can be utilized for multiple purposes. While GANs have demonstrated tremendous successes in producing synthetic data samples that replicate the dynamics of the original datasets, the validity of the synthetic data and the underlying privacy concerns represent major challenges which are not sufficiently addressed. In this work, we design a cascaded tabular GAN framework (CasTGAN) for generating realistic tabular data with a specific focus on the validity of the output. In this context, validity refers to the the dependency between features that can be found in the real data, but is typically misrepresented by traditional generative models. Our key idea entails that employing a cascaded architecture in which a dedicated generator samples each feature, the synthetic output becomes more representative of the real data. Our experimental results demonstrate that our model well captures the constraints and the correlations between the features of the real data, especially the high dimensional datasets. Furthermore, we evaluate the risk of white-box privacy attacks on our model and subsequently show that applying some perturbations to the auxiliary learners in CasTGAN increases the overall robustness of our model against targeted attacks.



## **39. A First Order Meta Stackelberg Method for Robust Federated Learning (Technical Report)**

cs.CR

Accepted to ICML 2023 Workshop on The 2nd New Frontiers In  Adversarial Machine Learning. Workshop Proceedings version: arXiv:2306.13800

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2306.13273v2) [paper-pdf](http://arxiv.org/pdf/2306.13273v2)

**Authors**: Henger Li, Tianyi Xu, Tao Li, Yunian Pan, Quanyan Zhu, Zizhan Zheng

**Abstract**: Recent research efforts indicate that federated learning (FL) systems are vulnerable to a variety of security breaches. While numerous defense strategies have been suggested, they are mainly designed to counter specific attack patterns and lack adaptability, rendering them less effective when facing uncertain or adaptive threats. This work models adversarial FL as a Bayesian Stackelberg Markov game (BSMG) between the defender and the attacker to address the lack of adaptability to uncertain adaptive attacks. We further devise an effective meta-learning technique to solve for the Stackelberg equilibrium, leading to a resilient and adaptable defense. The experiment results suggest that our meta-Stackelberg learning approach excels in combating intense model poisoning and backdoor attacks of indeterminate types.



## **40. A First Order Meta Stackelberg Method for Robust Federated Learning**

cs.LG

Accepted to ICML 2023 Workshop on The 2nd New Frontiers In  Adversarial Machine Learning. Associated technical report arXiv:2306.13273

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2306.13800v2) [paper-pdf](http://arxiv.org/pdf/2306.13800v2)

**Authors**: Yunian Pan, Tao Li, Henger Li, Tianyi Xu, Zizhan Zheng, Quanyan Zhu

**Abstract**: Previous research has shown that federated learning (FL) systems are exposed to an array of security risks. Despite the proposal of several defensive strategies, they tend to be non-adaptive and specific to certain types of attacks, rendering them ineffective against unpredictable or adaptive threats. This work models adversarial federated learning as a Bayesian Stackelberg Markov game (BSMG) to capture the defender's incomplete information of various attack types. We propose meta-Stackelberg learning (meta-SL), a provably efficient meta-learning algorithm, to solve the equilibrium strategy in BSMG, leading to an adaptable FL defense. We demonstrate that meta-SL converges to the first-order $\varepsilon$-equilibrium point in $O(\varepsilon^{-2})$ gradient iterations, with $O(\varepsilon^{-4})$ samples needed per iteration, matching the state of the art. Empirical evidence indicates that our meta-Stackelberg framework performs exceptionally well against potent model poisoning and backdoor attacks of an uncertain nature.



## **41. Fedward: Flexible Federated Backdoor Defense Framework with Non-IID Data**

cs.LG

Accepted by IEEE ICME 2023

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00356v1) [paper-pdf](http://arxiv.org/pdf/2307.00356v1)

**Authors**: Zekai Chen, Fuyi Wang, Zhiwei Zheng, Ximeng Liu, Yujie Lin

**Abstract**: Federated learning (FL) enables multiple clients to collaboratively train deep learning models while considering sensitive local datasets' privacy. However, adversaries can manipulate datasets and upload models by injecting triggers for federated backdoor attacks (FBA). Existing defense strategies against FBA consider specific and limited attacker models, and a sufficient amount of noise to be injected only mitigates rather than eliminates FBA. To address these deficiencies, we introduce a Flexible Federated Backdoor Defense Framework (Fedward) to ensure the elimination of adversarial backdoors. We decompose FBA into various attacks, and design amplified magnitude sparsification (AmGrad) and adaptive OPTICS clustering (AutoOPTICS) to address each attack. Meanwhile, Fedward uses the adaptive clipping method by regarding the number of samples in the benign group as constraints on the boundary. This ensures that Fedward can maintain the performance for the Non-IID scenario. We conduct experimental evaluations over three benchmark datasets and thoroughly compare them to state-of-the-art studies. The results demonstrate the promising defense performance from Fedward, moderately improved by 33% $\sim$ 75 in clustering defense methods, and 96.98%, 90.74%, and 89.8% for Non-IID to the utmost extent for the average FBA success rate over MNIST, FMNIST, and CIFAR10, respectively.



## **42. Adversarial Attacks and Defenses on 3D Point Cloud Classification: A Survey**

cs.CV

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00309v1) [paper-pdf](http://arxiv.org/pdf/2307.00309v1)

**Authors**: Hanieh Naderi, Ivan V. Bajić

**Abstract**: Deep learning has successfully solved a wide range of tasks in 2D vision as a dominant AI technique. Recently, deep learning on 3D point clouds is becoming increasingly popular for addressing various tasks in this field. Despite remarkable achievements, deep learning algorithms are vulnerable to adversarial attacks. These attacks are imperceptible to the human eye but can easily fool deep neural networks in the testing and deployment stage. To encourage future research, this survey summarizes the current progress on adversarial attack and defense techniques on point cloud classification. This paper first introduces the principles and characteristics of adversarial attacks and summarizes and analyzes the adversarial example generation methods in recent years. Besides, it classifies defense strategies as input transformation, data optimization, and deep model modification. Finally, it presents several challenging issues and future research directions in this domain.



## **43. Common Knowledge Learning for Generating Transferable Adversarial Examples**

cs.LG

11 pages, 5 figures

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00274v1) [paper-pdf](http://arxiv.org/pdf/2307.00274v1)

**Authors**: Ruijie Yang, Yuanfang Guo, Junfu Wang, Jiantao Zhou, Yunhong Wang

**Abstract**: This paper focuses on an important type of black-box attacks, i.e., transfer-based adversarial attacks, where the adversary generates adversarial examples by a substitute (source) model and utilize them to attack an unseen target model, without knowing its information. Existing methods tend to give unsatisfactory adversarial transferability when the source and target models are from different types of DNN architectures (e.g. ResNet-18 and Swin Transformer). In this paper, we observe that the above phenomenon is induced by the output inconsistency problem. To alleviate this problem while effectively utilizing the existing DNN models, we propose a common knowledge learning (CKL) framework to learn better network weights to generate adversarial examples with better transferability, under fixed network architectures. Specifically, to reduce the model-specific features and obtain better output distributions, we construct a multi-teacher framework, where the knowledge is distilled from different teacher architectures into one student network. By considering that the gradient of input is usually utilized to generated adversarial examples, we impose constraints on the gradients between the student and teacher models, to further alleviate the output inconsistency problem and enhance the adversarial transferability. Extensive experiments demonstrate that our proposed work can significantly improve the adversarial transferability.



## **44. Hiding in Plain Sight: Differential Privacy Noise Exploitation for Evasion-resilient Localized Poisoning Attacks in Multiagent Reinforcement Learning**

cs.LG

Accepted for publication in the proceeding of ICMLC 2023, 9-11 July  2023, The University of Adelaide, Adelaide, Australia

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00268v1) [paper-pdf](http://arxiv.org/pdf/2307.00268v1)

**Authors**: Md Tamjid Hossain, Hung La

**Abstract**: Lately, differential privacy (DP) has been introduced in cooperative multiagent reinforcement learning (CMARL) to safeguard the agents' privacy against adversarial inference during knowledge sharing. Nevertheless, we argue that the noise introduced by DP mechanisms may inadvertently give rise to a novel poisoning threat, specifically in the context of private knowledge sharing during CMARL, which remains unexplored in the literature. To address this shortcoming, we present an adaptive, privacy-exploiting, and evasion-resilient localized poisoning attack (PeLPA) that capitalizes on the inherent DP-noise to circumvent anomaly detection systems and hinder the optimal convergence of the CMARL model. We rigorously evaluate our proposed PeLPA attack in diverse environments, encompassing both non-adversarial and multiple-adversarial contexts. Our findings reveal that, in a medium-scale environment, the PeLPA attack with attacker ratios of 20% and 40% can lead to an increase in average steps to goal by 50.69% and 64.41%, respectively. Furthermore, under similar conditions, PeLPA can result in a 1.4x and 1.6x computational time increase in optimal reward attainment and a 1.18x and 1.38x slower convergence for attacker ratios of 20% and 40%, respectively.



## **45. A Black-box NLP Classifier Attacker**

cs.LG

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2112.11660v3) [paper-pdf](http://arxiv.org/pdf/2112.11660v3)

**Authors**: Yueyang Liu, Hunmin Lee, Zhipeng Cai

**Abstract**: Deep neural networks have a wide range of applications in solving various real-world tasks and have achieved satisfactory results, in domains such as computer vision, image classification, and natural language processing. Meanwhile, the security and robustness of neural networks have become imperative, as diverse researches have shown the vulnerable aspects of neural networks. Case in point, in Natural language processing tasks, the neural network may be fooled by an attentively modified text, which has a high similarity to the original one. As per previous research, most of the studies are focused on the image domain; Different from image adversarial attacks, the text is represented in a discrete sequence, traditional image attack methods are not applicable in the NLP field. In this paper, we propose a word-level NLP sentiment classifier attack model, which includes a self-attention mechanism-based word selection method and a greedy search algorithm for word substitution. We experiment with our attack model by attacking GRU and 1D-CNN victim models on IMDB datasets. Experimental results demonstrate that our model achieves a higher attack success rate and more efficient than previous methods due to the efficient word selection algorithms are employed and minimized the word substitute number. Also, our model is transferable, which can be used in the image domain with several modifications.



## **46. SecBeam: Securing mmWave Beam Alignment against Beam-Stealing Attacks**

cs.CR

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00178v1) [paper-pdf](http://arxiv.org/pdf/2307.00178v1)

**Authors**: Jingcheng Li, Loukas Lazos, Ming Li

**Abstract**: Millimeter wave (mmWave) communications employ narrow-beam directional communications to compensate for the high path loss at mmWave frequencies. Compared to their omnidirectional counterparts, an additional step of aligning the transmitter's and receiver's antennas is required. In current standards such as 802.11ad, this beam alignment process is implemented via an exhaustive search through the horizontal plane known as beam sweeping. However, the beam sweeping process is unauthenticated. As a result, an adversary, Mallory, can launch an active beam-stealing attack by injecting forged beacons of high power, forcing the legitimate devices to beamform towards her direction. Mallory is now in control of the communication link between the two devices, thus breaking the false sense of security given by the directionality of mmWave transmissions.   Prior works have added integrity protection to beam alignment messages to prevent forgeries. In this paper, we demonstrate a new beam-stealing attack that does not require message forging. We show that Mallory can amplify and relay a beam sweeping frame from her direction without altering its contents. Intuitively, cryptographic primitives cannot verify physical properties such as the SNR used in beam selection. We propose a new beam sweeping protocol called SecBeam that utilizes power/sector randomization and coarse angle-of-arrival information to detect amplify-and-relay attacks. We demonstrate the security and performance of SecBeam using an experimental mmWave platform and via ray-tracing simulations.



## **47. Beyond Neural-on-Neural Approaches to Speaker Gender Protection**

eess.AS

**SubmitDate**: 2023-06-30    [abs](http://arxiv.org/abs/2306.17700v1) [paper-pdf](http://arxiv.org/pdf/2306.17700v1)

**Authors**: Loes van Bemmel, Zhuoran Liu, Nik Vaessen, Martha Larson

**Abstract**: Recent research has proposed approaches that modify speech to defend against gender inference attacks. The goal of these protection algorithms is to control the availability of information about a speaker's gender, a privacy-sensitive attribute. Currently, the common practice for developing and testing gender protection algorithms is "neural-on-neural", i.e., perturbations are generated and tested with a neural network. In this paper, we propose to go beyond this practice to strengthen the study of gender protection. First, we demonstrate the importance of testing gender inference attacks that are based on speech features historically developed by speech scientists, alongside the conventionally used neural classifiers. Next, we argue that researchers should use speech features to gain insight into how protective modifications change the speech signal. Finally, we point out that gender-protection algorithms should be compared with novel "vocal adversaries", human-executed voice adaptations, in order to improve interpretability and enable before-the-mic protection.



## **48. MalProtect: Stateful Defense Against Adversarial Query Attacks in ML-based Malware Detection**

cs.LG

**SubmitDate**: 2023-06-30    [abs](http://arxiv.org/abs/2302.10739v3) [paper-pdf](http://arxiv.org/pdf/2302.10739v3)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: ML models are known to be vulnerable to adversarial query attacks. In these attacks, queries are iteratively perturbed towards a particular class without any knowledge of the target model besides its output. The prevalence of remotely-hosted ML classification models and Machine-Learning-as-a-Service platforms means that query attacks pose a real threat to the security of these systems. To deal with this, stateful defenses have been proposed to detect query attacks and prevent the generation of adversarial examples by monitoring and analyzing the sequence of queries received by the system. Several stateful defenses have been proposed in recent years. However, these defenses rely solely on similarity or out-of-distribution detection methods that may be effective in other domains. In the malware detection domain, the methods to generate adversarial examples are inherently different, and therefore we find that such detection mechanisms are significantly less effective. Hence, in this paper, we present MalProtect, which is a stateful defense against query attacks in the malware detection domain. MalProtect uses several threat indicators to detect attacks. Our results show that it reduces the evasion rate of adversarial query attacks by 80+\% in Android and Windows malware, across a range of attacker scenarios. In the first evaluation of its kind, we show that MalProtect outperforms prior stateful defenses, especially under the peak adversarial threat.



## **49. Efficient Backdoor Removal Through Natural Gradient Fine-tuning**

cs.CV

**SubmitDate**: 2023-06-30    [abs](http://arxiv.org/abs/2306.17441v1) [paper-pdf](http://arxiv.org/pdf/2306.17441v1)

**Authors**: Nazmul Karim, Abdullah Al Arafat, Umar Khalid, Zhishan Guo, Naznin Rahnavard

**Abstract**: The success of a deep neural network (DNN) heavily relies on the details of the training scheme; e.g., training data, architectures, hyper-parameters, etc. Recent backdoor attacks suggest that an adversary can take advantage of such training details and compromise the integrity of a DNN. Our studies show that a backdoor model is usually optimized to a bad local minima, i.e. sharper minima as compared to a benign model. Intuitively, a backdoor model can be purified by reoptimizing the model to a smoother minima through fine-tuning with a few clean validation data. However, fine-tuning all DNN parameters often requires huge computational costs and often results in sub-par clean test performance. To address this concern, we propose a novel backdoor purification technique, Natural Gradient Fine-tuning (NGF), which focuses on removing the backdoor by fine-tuning only one layer. Specifically, NGF utilizes a loss surface geometry-aware optimizer that can successfully overcome the challenge of reaching a smooth minima under a one-layer optimization scenario. To enhance the generalization performance of our proposed method, we introduce a clean data distribution-aware regularizer based on the knowledge of loss surface curvature matrix, i.e., Fisher Information Matrix. Extensive experiments show that the proposed method achieves state-of-the-art performance on a wide range of backdoor defense benchmarks: four different datasets- CIFAR10, GTSRB, Tiny-ImageNet, and ImageNet; 13 recent backdoor attacks, e.g. Blend, Dynamic, WaNet, ISSBA, etc.



## **50. LTD: Low Temperature Distillation for Robust Adversarial Training**

cs.CV

**SubmitDate**: 2023-06-30    [abs](http://arxiv.org/abs/2111.02331v3) [paper-pdf](http://arxiv.org/pdf/2111.02331v3)

**Authors**: Erh-Chung Chen, Che-Rung Lee

**Abstract**: Adversarial training has been widely used to enhance the robustness of neural network models against adversarial attacks. Despite the popularity of neural network models, a significant gap exists between the natural and robust accuracy of these models. In this paper, we identify one of the primary reasons for this gap is the common use of one-hot vectors as labels, which hinders the learning process for image recognition. Representing ambiguous images with one-hot vectors is imprecise and may lead the model to suboptimal solutions. To overcome this issue, we propose a novel method called Low Temperature Distillation (LTD) that generates soft labels using the modified knowledge distillation framework. Unlike previous approaches, LTD uses a relatively low temperature in the teacher model and fixed, but different temperatures for the teacher and student models. This modification boosts the model's robustness without encountering the gradient masking problem that has been addressed in defensive distillation. The experimental results demonstrate the effectiveness of the proposed LTD method combined with previous techniques, achieving robust accuracy rates of 58.19%, 31.13%, and 42.08% on CIFAR-10, CIFAR-100, and ImageNet data sets, respectively, without additional unlabeled data.



