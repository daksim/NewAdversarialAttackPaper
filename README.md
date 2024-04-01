# Latest Adversarial Attack Papers
**update at 2024-04-01 09:32:19**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Selective Attention-based Modulation for Continual Learning**

cs.CV

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2403.20086v1) [paper-pdf](http://arxiv.org/pdf/2403.20086v1)

**Authors**: Giovanni Bellitto, Federica Proietto Salanitri, Matteo Pennisi, Matteo Boschini, Angelo Porrello, Simone Calderara, Simone Palazzo, Concetto Spampinato

**Abstract**: We present SAM, a biologically-plausible selective attention-driven modulation approach to enhance classification models in a continual learning setting. Inspired by neurophysiological evidence that the primary visual cortex does not contribute to object manifold untangling for categorization and that primordial attention biases are still embedded in the modern brain, we propose to employ auxiliary saliency prediction features as a modulation signal to drive and stabilize the learning of a sequence of non-i.i.d. classification tasks. Experimental results confirm that SAM effectively enhances the performance (in some cases up to about twenty percent points) of state-of-the-art continual learning methods, both in class-incremental and task-incremental settings. Moreover, we show that attention-based modulation successfully encourages the learning of features that are more robust to the presence of spurious features and to adversarial attacks than baseline methods. Code is available at: https://github.com/perceivelab/SAM.



## **2. Strong Transferable Adversarial Attacks via Ensembled Asymptotically Normal Distribution Learning**

cs.LG

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2209.11964v2) [paper-pdf](http://arxiv.org/pdf/2209.11964v2)

**Authors**: Zhengwei Fang, Rui Wang, Tao Huang, Liping Jing

**Abstract**: Strong adversarial examples are crucial for evaluating and enhancing the robustness of deep neural networks. However, the performance of popular attacks is usually sensitive, for instance, to minor image transformations, stemming from limited information -- typically only one input example, a handful of white-box source models, and undefined defense strategies. Hence, the crafted adversarial examples are prone to overfit the source model, which hampers their transferability to unknown architectures. In this paper, we propose an approach named Multiple Asymptotically Normal Distribution Attacks (MultiANDA) which explicitly characterize adversarial perturbations from a learned distribution. Specifically, we approximate the posterior distribution over the perturbations by taking advantage of the asymptotic normality property of stochastic gradient ascent (SGA), then employ the deep ensemble strategy as an effective proxy for Bayesian marginalization in this process, aiming to estimate a mixture of Gaussians that facilitates a more thorough exploration of the potential optimization space. The approximated posterior essentially describes the stationary distribution of SGA iterations, which captures the geometric information around the local optimum. Thus, MultiANDA allows drawing an unlimited number of adversarial perturbations for each input and reliably maintains the transferability. Our proposed method outperforms ten state-of-the-art black-box attacks on deep learning models with or without defenses through extensive experiments on seven normally trained and seven defense models.



## **3. An Anomaly Behavior Analysis Framework for Securing Autonomous Vehicle Perception**

cs.RO

20th ACS/IEEE International Conference on Computer Systems and  Applications (Accepted for publication)

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2310.05041v2) [paper-pdf](http://arxiv.org/pdf/2310.05041v2)

**Authors**: Murad Mehrab Abrar, Salim Hariri

**Abstract**: As a rapidly growing cyber-physical platform, Autonomous Vehicles (AVs) are encountering more security challenges as their capabilities continue to expand. In recent years, adversaries are actively targeting the perception sensors of autonomous vehicles with sophisticated attacks that are not easily detected by the vehicles' control systems. This work proposes an Anomaly Behavior Analysis approach to detect a perception sensor attack against an autonomous vehicle. The framework relies on temporal features extracted from a physics-based autonomous vehicle behavior model to capture the normal behavior of vehicular perception in autonomous driving. By employing a combination of model-based techniques and machine learning algorithms, the proposed framework distinguishes between normal and abnormal vehicular perception behavior. To demonstrate the application of the framework in practice, we performed a depth camera attack experiment on an autonomous vehicle testbed and generated an extensive dataset. We validated the effectiveness of the proposed framework using this real-world data and released the dataset for public access. To our knowledge, this dataset is the first of its kind and will serve as a valuable resource for the research community in evaluating their intrusion detection techniques effectively.



## **4. MMCert: Provable Defense against Adversarial Attacks to Multi-modal Models**

cs.CV

To appear in CVPR'24

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2403.19080v2) [paper-pdf](http://arxiv.org/pdf/2403.19080v2)

**Authors**: Yanting Wang, Hongye Fu, Wei Zou, Jinyuan Jia

**Abstract**: Different from a unimodal model whose input is from a single modality, the input (called multi-modal input) of a multi-modal model is from multiple modalities such as image, 3D points, audio, text, etc. Similar to unimodal models, many existing studies show that a multi-modal model is also vulnerable to adversarial perturbation, where an attacker could add small perturbation to all modalities of a multi-modal input such that the multi-modal model makes incorrect predictions for it. Existing certified defenses are mostly designed for unimodal models, which achieve sub-optimal certified robustness guarantees when extended to multi-modal models as shown in our experimental results. In our work, we propose MMCert, the first certified defense against adversarial attacks to a multi-modal model. We derive a lower bound on the performance of our MMCert under arbitrary adversarial attacks with bounded perturbations to both modalities (e.g., in the context of auto-driving, we bound the number of changed pixels in both RGB image and depth image). We evaluate our MMCert using two benchmark datasets: one for the multi-modal road segmentation task and the other for the multi-modal emotion recognition task. Moreover, we compare our MMCert with a state-of-the-art certified defense extended from unimodal models. Our experimental results show that our MMCert outperforms the baseline.



## **5. Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy**

cs.LG

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.16591v2) [paper-pdf](http://arxiv.org/pdf/2403.16591v2)

**Authors**: Xiaojin Zhang, Yulin Fei, Wei Chen, Hai Jin

**Abstract**: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between LDP and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. The relationship between LDP and Maximum Bayesian Privacy (MBP) is first revealed, demonstrating that under uniform prior distribution, a mechanism satisfying $\xi$-LDP will satisfy $\xi$-MBP and conversely $\xi$-MBP also confers 2$\xi$-LDP. Our next theoretical contribution are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Maximum Bayesian Privacy (MBP), encapsulated by equations $\epsilon_{p,a} \leq \frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p,m} + \epsilon)\cdot(e^{\epsilon_{p,m} + \epsilon} - 1)}$. These relationships fortify our understanding of the privacy guarantees provided by various mechanisms. Our work not only lays the groundwork for future empirical exploration but also promises to facilitate the design of privacy-preserving algorithms, thereby fostering the development of trustworthy machine learning solutions.



## **6. Evolving Assembly Code in an Adversarial Environment**

cs.NE

9 pages, 5 figures, 6 listings

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19489v1) [paper-pdf](http://arxiv.org/pdf/2403.19489v1)

**Authors**: Irina Maliukov, Gera Weiss, Oded Margalit, Achiya Elyasaf

**Abstract**: In this work, we evolve assembly code for the CodeGuru competition. The competition's goal is to create a survivor -- an assembly program that runs the longest in shared memory, by resisting attacks from adversary survivors and finding their weaknesses. For evolving top-notch solvers, we specify a Backus Normal Form (BNF) for the assembly language and synthesize the code from scratch using Genetic Programming (GP). We evaluate the survivors by running CodeGuru games against human-written winning survivors. Our evolved programs found weaknesses in the programs they were trained against and utilized them. In addition, we compare our approach with a Large-Language Model, demonstrating that the latter cannot generate a survivor that can win at any competition. This work has important applications for cyber-security, as we utilize evolution to detect weaknesses in survivors. The assembly BNF is domain-independent; thus, by modifying the fitness function, it can detect code weaknesses and help fix them. Finally, the CodeGuru competition offers a novel platform for analyzing GP and code evolution in adversarial environments. To support further research in this direction, we provide a thorough qualitative analysis of the evolved survivors and the weaknesses found.



## **7. Cloudy with a Chance of Cyberattacks: Dangling Resources Abuse on Cloud Platforms**

cs.NI

17 pages, 29 figures, to be published in NSDI'24: Proceedings of the  21st USENIX Symposium on Networked Systems Design and Implementation

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19368v1) [paper-pdf](http://arxiv.org/pdf/2403.19368v1)

**Authors**: Jens Frieß, Tobias Gattermayer, Nethanel Gelernter, Haya Schulmann, Michael Waidner

**Abstract**: Recent works showed that it is feasible to hijack resources on cloud platforms. In such hijacks, attackers can take over released resources that belong to legitimate organizations. It was proposed that adversaries could abuse these resources to carry out attacks against customers of the hijacked services, e.g., through malware distribution. However, to date, no research has confirmed the existence of these attacks. We identify, for the first time, real-life hijacks of cloud resources. This yields a number of surprising and important insights. First, contrary to previous assumption that attackers primarily target IP addresses, our findings reveal that the type of resource is not the main consideration in a hijack. Attackers focus on hijacking records that allow them to determine the resource by entering freetext. The costs and overhead of hijacking such records are much lower than those of hijacking IP addresses, which are randomly selected from a large pool. Second, identifying hijacks poses a substantial challenge. Monitoring resource changes, e.g., changes in content, is insufficient, since such changes could also be legitimate. Retrospective analysis of digital assets to identify hijacks is also arduous due to the immense volume of data involved and the absence of indicators to search for. To address this challenge, we develop a novel approach that involves analyzing data from diverse sources to effectively differentiate between malicious and legitimate modifications. Our analysis has revealed 20,904 instances of hijacked resources on popular cloud platforms. While some hijacks are short-lived (up to 15 days), 1/3 persist for more than 65 days. We study how attackers abuse the hijacked resources and find that, in contrast to the threats considered in previous work, the majority of the abuse (75%) is blackhat search engine optimization.



## **8. Data-free Defense of Black Box Models Against Adversarial Attacks**

cs.LG

CVPR Workshop (Under Review)

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2211.01579v3) [paper-pdf](http://arxiv.org/pdf/2211.01579v3)

**Authors**: Gaurav Kumar Nayak, Inder Khatri, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Several companies often safeguard their trained deep models (i.e., details of architecture, learnt weights, training details etc.) from third-party users by exposing them only as black boxes through APIs. Moreover, they may not even provide access to the training data due to proprietary reasons or sensitivity concerns. In this work, we propose a novel defense mechanism for black box models against adversarial attacks in a data-free set up. We construct synthetic data via generative model and train surrogate network using model stealing techniques. To minimize adversarial contamination on perturbed samples, we propose 'wavelet noise remover' (WNR) that performs discrete wavelet decomposition on input images and carefully select only a few important coefficients determined by our 'wavelet coefficient selection module' (WCSM). To recover the high-frequency content of the image after noise removal via WNR, we further train a 'regenerator' network with an objective to retrieve the coefficients such that the reconstructed image yields similar to original predictions on the surrogate model. At test time, WNR combined with trained regenerator network is prepended to the black box network, resulting in a high boost in adversarial accuracy. Our method improves the adversarial accuracy on CIFAR-10 by 38.98% and 32.01% on state-of-the-art Auto Attack compared to baseline, even when the attacker uses surrogate architecture (Alexnet-half and Alexnet) similar to the black box architecture (Alexnet) with same model stealing strategy as defender. The code is available at https://github.com/vcl-iisc/data-free-black-box-defense



## **9. Feature Unlearning for Pre-trained GANs and VAEs**

cs.CV

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2303.05699v4) [paper-pdf](http://arxiv.org/pdf/2303.05699v4)

**Authors**: Saemi Moon, Seunghyuk Cho, Dongwoo Kim

**Abstract**: We tackle the problem of feature unlearning from a pre-trained image generative model: GANs and VAEs. Unlike a common unlearning task where an unlearning target is a subset of the training set, we aim to unlearn a specific feature, such as hairstyle from facial images, from the pre-trained generative models. As the target feature is only presented in a local region of an image, unlearning the entire image from the pre-trained model may result in losing other details in the remaining region of the image. To specify which features to unlearn, we collect randomly generated images that contain the target features. We then identify a latent representation corresponding to the target feature and then use the representation to fine-tune the pre-trained model. Through experiments on MNIST, CelebA, and FFHQ datasets, we show that target features are successfully removed while keeping the fidelity of the original models. Further experiments with an adversarial attack show that the unlearned model is more robust under the presence of malicious parties.



## **10. Data Poisoning for In-context Learning**

cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2402.02160v2) [paper-pdf](http://arxiv.org/pdf/2402.02160v2)

**Authors**: Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, Jiliang Tang

**Abstract**: In the domain of large language models (LLMs), in-context learning (ICL) has been recognized for its innovative ability to adapt to new tasks, relying on examples rather than retraining or fine-tuning. This paper delves into the critical issue of ICL's susceptibility to data poisoning attacks, an area not yet fully explored. We wonder whether ICL is vulnerable, with adversaries capable of manipulating example data to degrade model performance. To address this, we introduce ICLPoison, a specialized attacking framework conceived to exploit the learning mechanisms of ICL. Our approach uniquely employs discrete text perturbations to strategically influence the hidden states of LLMs during the ICL process. We outline three representative strategies to implement attacks under our framework, each rigorously evaluated across a variety of models and tasks. Our comprehensive tests, including trials on the sophisticated GPT-4 model, demonstrate that ICL's performance is significantly compromised under our framework. These revelations indicate an urgent need for enhanced defense mechanisms to safeguard the integrity and reliability of LLMs in applications relying on in-context learning.



## **11. Towards Sustainable SecureML: Quantifying Carbon Footprint of Adversarial Machine Learning**

cs.LG

Accepted at GreenNet Workshop @ IEEE International Conference on  Communications (IEEE ICC 2024)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.19009v1) [paper-pdf](http://arxiv.org/pdf/2403.19009v1)

**Authors**: Syed Mhamudul Hasan, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The widespread adoption of machine learning (ML) across various industries has raised sustainability concerns due to its substantial energy usage and carbon emissions. This issue becomes more pressing in adversarial ML, which focuses on enhancing model security against different network-based attacks. Implementing defenses in ML systems often necessitates additional computational resources and network security measures, exacerbating their environmental impacts. In this paper, we pioneer the first investigation into adversarial ML's carbon footprint, providing empirical evidence connecting greater model robustness to higher emissions. Addressing the critical need to quantify this trade-off, we introduce the Robustness Carbon Trade-off Index (RCTI). This novel metric, inspired by economic elasticity principles, captures the sensitivity of carbon emissions to changes in adversarial robustness. We demonstrate the RCTI through an experiment involving evasion attacks, analyzing the interplay between robustness against attacks, performance, and carbon emissions.



## **12. Robustness and Visual Explanation for Black Box Image, Video, and ECG Signal Classification with Reinforcement Learning**

cs.LG

AAAI Proceedings reference:  https://ojs.aaai.org/index.php/AAAI/article/view/30579

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18985v1) [paper-pdf](http://arxiv.org/pdf/2403.18985v1)

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Avisek Naug, Sahand Ghorbanpour

**Abstract**: We present a generic Reinforcement Learning (RL) framework optimized for crafting adversarial attacks on different model types spanning from ECG signal analysis (1D), image classification (2D), and video classification (3D). The framework focuses on identifying sensitive regions and inducing misclassifications with minimal distortions and various distortion types. The novel RL method outperforms state-of-the-art methods for all three applications, proving its efficiency. Our RL approach produces superior localization masks, enhancing interpretability for image classification and ECG analysis models. For applications such as ECG analysis, our platform highlights critical ECG segments for clinicians while ensuring resilience against prevalent distortions. This comprehensive tool aims to bolster both resilience with adversarial training and transparency across varied applications and data types.



## **13. Deep Learning for Robust and Explainable Models in Computer Vision**

cs.CV

150 pages, 37 figures, 12 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18674v1) [paper-pdf](http://arxiv.org/pdf/2403.18674v1)

**Authors**: Mohammadreza Amirian

**Abstract**: Recent breakthroughs in machine and deep learning (ML and DL) research have provided excellent tools for leveraging enormous amounts of data and optimizing huge models with millions of parameters to obtain accurate networks for image processing. These developments open up tremendous opportunities for using artificial intelligence (AI) in the automation and human assisted AI industry. However, as more and more models are deployed and used in practice, many challenges have emerged. This thesis presents various approaches that address robustness and explainability challenges for using ML and DL in practice.   Robustness and reliability are the critical components of any model before certification and deployment in practice. Deep convolutional neural networks (CNNs) exhibit vulnerability to transformations of their inputs, such as rotation and scaling, or intentional manipulations as described in the adversarial attack literature. In addition, building trust in AI-based models requires a better understanding of current models and developing methods that are more explainable and interpretable a priori.   This thesis presents developments in computer vision models' robustness and explainability. Furthermore, this thesis offers an example of using vision models' feature response visualization (models' interpretations) to improve robustness despite interpretability and robustness being seemingly unrelated in the related research. Besides methodological developments for robust and explainable vision models, a key message of this thesis is introducing model interpretation techniques as a tool for understanding vision models and improving their design and robustness. In addition to the theoretical developments, this thesis demonstrates several applications of ML and DL in different contexts, such as medical imaging and affective computing.



## **14. LCANets++: Robust Audio Classification using Multi-layer Neural Networks with Lateral Competition**

cs.SD

Accepted at 2024 IEEE International Conference on Acoustics, Speech  and Signal Processing Workshops (ICASSPW)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2308.12882v2) [paper-pdf](http://arxiv.org/pdf/2308.12882v2)

**Authors**: Sayanton V. Dibbo, Juston S. Moore, Garrett T. Kenyon, Michael A. Teti

**Abstract**: Audio classification aims at recognizing audio signals, including speech commands or sound events. However, current audio classifiers are susceptible to perturbations and adversarial attacks. In addition, real-world audio classification tasks often suffer from limited labeled data. To help bridge these gaps, previous work developed neuro-inspired convolutional neural networks (CNNs) with sparse coding via the Locally Competitive Algorithm (LCA) in the first layer (i.e., LCANets) for computer vision. LCANets learn in a combination of supervised and unsupervised learning, reducing dependency on labeled samples. Motivated by the fact that auditory cortex is also sparse, we extend LCANets to audio recognition tasks and introduce LCANets++, which are CNNs that perform sparse coding in multiple layers via LCA. We demonstrate that LCANets++ are more robust than standard CNNs and LCANets against perturbations, e.g., background noise, as well as black-box and white-box attacks, e.g., evasion and fast gradient sign (FGSM) attacks.



## **15. The Impact of Uniform Inputs on Activation Sparsity and Energy-Latency Attacks in Computer Vision**

cs.CR

Accepted at the DLSP 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18587v1) [paper-pdf](http://arxiv.org/pdf/2403.18587v1)

**Authors**: Andreas Müller, Erwin Quiring

**Abstract**: Resource efficiency plays an important role for machine learning nowadays. The energy and decision latency are two critical aspects to ensure a sustainable and practical application. Unfortunately, the energy consumption and decision latency are not robust against adversaries. Researchers have recently demonstrated that attackers can compute and submit so-called sponge examples at inference time to increase the energy consumption and decision latency of neural networks. In computer vision, the proposed strategy crafts inputs with less activation sparsity which could otherwise be used to accelerate the computation. In this paper, we analyze the mechanism how these energy-latency attacks reduce activation sparsity. In particular, we find that input uniformity is a key enabler. A uniform image, that is, an image with mostly flat, uniformly colored surfaces, triggers more activations due to a specific interplay of convolution, batch normalization, and ReLU activation. Based on these insights, we propose two new simple, yet effective strategies for crafting sponge examples: sampling images from a probability distribution and identifying dense, yet inconspicuous inputs in natural datasets. We empirically examine our findings in a comprehensive evaluation with multiple image classification models and show that our attack achieves the same sparsity effect as prior sponge-example methods, but at a fraction of computation effort. We also show that our sponge examples transfer between different neural networks. Finally, we discuss applications of our findings for the good by improving efficiency by increasing sparsity.



## **16. MisGUIDE : Defense Against Data-Free Deep Learning Model Extraction**

cs.CR

Under Review

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18580v1) [paper-pdf](http://arxiv.org/pdf/2403.18580v1)

**Authors**: Mahendra Gurve, Sankar Behera, Satyadev Ahlawat, Yamuna Prasad

**Abstract**: The rise of Machine Learning as a Service (MLaaS) has led to the widespread deployment of machine learning models trained on diverse datasets. These models are employed for predictive services through APIs, raising concerns about the security and confidentiality of the models due to emerging vulnerabilities in prediction APIs. Of particular concern are model cloning attacks, where individuals with limited data and no knowledge of the training dataset manage to replicate a victim model's functionality through black-box query access. This commonly entails generating adversarial queries to query the victim model, thereby creating a labeled dataset.   This paper proposes "MisGUIDE", a two-step defense framework for Deep Learning models that disrupts the adversarial sample generation process by providing a probabilistic response when the query is deemed OOD. The first step employs a Vision Transformer-based framework to identify OOD queries, while the second step perturbs the response for such queries, introducing a probabilistic loss function to MisGUIDE the attackers. The aim of the proposed defense method is to reduce the accuracy of the cloned model while maintaining accuracy on authentic queries. Extensive experiments conducted on two benchmark datasets demonstrate that the proposed framework significantly enhances the resistance against state-of-the-art data-free model extraction in black-box settings.



## **17. CosalPure: Learning Concept from Group Images for Robust Co-Saliency Detection**

cs.CV

8 pages

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18554v1) [paper-pdf](http://arxiv.org/pdf/2403.18554v1)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstract**: Co-salient object detection (CoSOD) aims to identify the common and salient (usually in the foreground) regions across a given group of images. Although achieving significant progress, state-of-the-art CoSODs could be easily affected by some adversarial perturbations, leading to substantial accuracy reduction. The adversarial perturbations can mislead CoSODs but do not change the high-level semantic information (e.g., concept) of the co-salient objects. In this paper, we propose a novel robustness enhancement framework by first learning the concept of the co-salient objects based on the input group images and then leveraging this concept to purify adversarial perturbations, which are subsequently fed to CoSODs for robustness enhancement. Specifically, we propose CosalPure containing two modules, i.e., group-image concept learning and concept-guided diffusion purification. For the first module, we adopt a pre-trained text-to-image diffusion model to learn the concept of co-salient objects within group images where the learned concept is robust to adversarial examples. For the second module, we map the adversarial image to the latent space and then perform diffusion generation by embedding the learned concept into the noise prediction function as an extra condition. Our method can effectively alleviate the influence of the SOTA adversarial attack containing different adversarial patterns, including exposure and noise. The extensive results demonstrate that our method could enhance the robustness of CoSODs significantly.



## **18. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2311.08268v3) [paper-pdf](http://arxiv.org/pdf/2311.08268v3)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.



## **19. Attack and Defense Analysis of Learned Image Compression**

eess.IV

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2401.10345v3) [paper-pdf](http://arxiv.org/pdf/2401.10345v3)

**Authors**: Tianyu Zhu, Heming Sun, Xiankui Xiong, Xuanpeng Zhu, Yong Gong, Minge jing, Yibo Fan

**Abstract**: Learned image compression (LIC) is becoming more and more popular these years with its high efficiency and outstanding compression quality. Still, the practicality against modified inputs added with specific noise could not be ignored. White-box attacks such as FGSM and PGD use only gradient to compute adversarial images that mislead LIC models to output unexpected results. Our experiments compare the effects of different dimensions such as attack methods, models, qualities, and targets, concluding that in the worst case, there is a 61.55% decrease in PSNR or a 19.15 times increase in bpp under the PGD attack. To improve their robustness, we conduct adversarial training by adding adversarial images into the training datasets, which obtains a 95.52% decrease in the R-D cost of the most vulnerable LIC model. We further test the robustness of H.266, whose better performance on reconstruction quality extends its possibility to defend one-step or iterative adversarial attacks.



## **20. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.16432v2) [paper-pdf](http://arxiv.org/pdf/2403.16432v2)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo.



## **21. SemRoDe: Macro Adversarial Training to Learn Representations That are Robust to Word-Level Attacks**

cs.CL

Published in NAACL 2024 (Main Track)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18423v1) [paper-pdf](http://arxiv.org/pdf/2403.18423v1)

**Authors**: Brian Formento, Wenjie Feng, Chuan Sheng Foo, Luu Anh Tuan, See-Kiong Ng

**Abstract**: Language models (LMs) are indispensable tools for natural language processing tasks, but their vulnerability to adversarial attacks remains a concern. While current research has explored adversarial training techniques, their improvements to defend against word-level attacks have been limited. In this work, we propose a novel approach called Semantic Robust Defence (SemRoDe), a Macro Adversarial Training strategy to enhance the robustness of LMs. Drawing inspiration from recent studies in the image domain, we investigate and later confirm that in a discrete data setting such as language, adversarial samples generated via word substitutions do indeed belong to an adversarial domain exhibiting a high Wasserstein distance from the base domain. Our method learns a robust representation that bridges these two domains. We hypothesize that if samples were not projected into an adversarial domain, but instead to a domain with minimal shift, it would improve attack robustness. We align the domains by incorporating a new distance-based objective. With this, our model is able to learn more generalized representations by aligning the model's high-level output features and therefore better handling unseen adversarial samples. This method can be generalized across word embeddings, even when they share minimal overlap at both vocabulary and word-substitution levels. To evaluate the effectiveness of our approach, we conduct experiments on BERT and RoBERTa models on three datasets. The results demonstrate promising state-of-the-art robustness.



## **22. LocalStyleFool: Regional Video Style Transfer Attack Using Segment Anything Model**

cs.CV

Accepted to 2024 IEEE Security and Privacy Workshops (SPW)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.11656v2) [paper-pdf](http://arxiv.org/pdf/2403.11656v2)

**Authors**: Yuxin Cao, Jinghao Li, Xi Xiao, Derui Wang, Minhui Xue, Hao Ge, Wei Liu, Guangwu Hu

**Abstract**: Previous work has shown that well-crafted adversarial perturbations can threaten the security of video recognition systems. Attackers can invade such models with a low query budget when the perturbations are semantic-invariant, such as StyleFool. Despite the query efficiency, the naturalness of the minutia areas still requires amelioration, since StyleFool leverages style transfer to all pixels in each frame. To close the gap, we propose LocalStyleFool, an improved black-box video adversarial attack that superimposes regional style-transfer-based perturbations on videos. Benefiting from the popularity and scalably usability of Segment Anything Model (SAM), we first extract different regions according to semantic information and then track them through the video stream to maintain the temporal consistency. Then, we add style-transfer-based perturbations to several regions selected based on the associative criterion of transfer-based gradient information and regional area. Perturbation fine adjustment is followed to make stylized videos adversarial. We demonstrate that LocalStyleFool can improve both intra-frame and inter-frame naturalness through a human-assessed survey, while maintaining competitive fooling rate and query efficiency. Successful experiments on the high-resolution dataset also showcase that scrupulous segmentation of SAM helps to improve the scalability of adversarial attacks under high-resolution data.



## **23. Physical 3D Adversarial Attacks against Monocular Depth Estimation in Autonomous Driving**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.17301v2) [paper-pdf](http://arxiv.org/pdf/2403.17301v2)

**Authors**: Junhao Zheng, Chenhao Lin, Jiahao Sun, Zhengyu Zhao, Qian Li, Chao Shen

**Abstract**: Deep learning-based monocular depth estimation (MDE), extensively applied in autonomous driving, is known to be vulnerable to adversarial attacks. Previous physical attacks against MDE models rely on 2D adversarial patches, so they only affect a small, localized region in the MDE map but fail under various viewpoints. To address these limitations, we propose 3D Depth Fool (3D$^2$Fool), the first 3D texture-based adversarial attack against MDE models. 3D$^2$Fool is specifically optimized to generate 3D adversarial textures agnostic to model types of vehicles and to have improved robustness in bad weather conditions, such as rain and fog. Experimental results validate the superior performance of our 3D$^2$Fool across various scenarios, including vehicles, MDE models, weather conditions, and viewpoints. Real-world experiments with printed 3D textures on physical vehicle models further demonstrate that our 3D$^2$Fool can cause an MDE error of over 10 meters.



## **24. Uncertainty-Aware SAR ATR: Defending Against Adversarial Attacks via Bayesian Neural Networks**

cs.CV

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18318v1) [paper-pdf](http://arxiv.org/pdf/2403.18318v1)

**Authors**: Tian Ye, Rajgopal Kannan, Viktor Prasanna, Carl Busart

**Abstract**: Adversarial attacks have demonstrated the vulnerability of Machine Learning (ML) image classifiers in Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) systems. An adversarial attack can deceive the classifier into making incorrect predictions by perturbing the input SAR images, for example, with a few scatterers attached to the on-ground objects. Therefore, it is critical to develop robust SAR ATR systems that can detect potential adversarial attacks by leveraging the inherent uncertainty in ML classifiers, thereby effectively alerting human decision-makers. In this paper, we propose a novel uncertainty-aware SAR ATR for detecting adversarial attacks. Specifically, we leverage the capability of Bayesian Neural Networks (BNNs) in performing image classification with quantified epistemic uncertainty to measure the confidence for each input SAR image. By evaluating the uncertainty, our method alerts when the input SAR image is likely to be adversarially generated. Simultaneously, we also generate visual explanations that reveal the specific regions in the SAR image where the adversarial scatterers are likely to to be present, thus aiding human decision-making with hints of evidence of adversarial attacks. Experiments on the MSTAR dataset demonstrate that our approach can identify over 80% adversarial SAR images with fewer than 20% false alarms, and our visual explanations can identify up to over 90% of scatterers in an adversarial SAR image.



## **25. Bayesian Learned Models Can Detect Adversarial Malware For Free**

cs.CR

Accepted to the 29th European Symposium on Research in Computer  Security (ESORICS) 2024 Conference

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18309v1) [paper-pdf](http://arxiv.org/pdf/2403.18309v1)

**Authors**: Bao Gia Doan, Dang Quang Nguyen, Paul Montague, Tamas Abraham, Olivier De Vel, Seyit Camtepe, Salil S. Kanhere, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: The vulnerability of machine learning-based malware detectors to adversarial attacks has prompted the need for robust solutions. Adversarial training is an effective method but is computationally expensive to scale up to large datasets and comes at the cost of sacrificing model performance for robustness. We hypothesize that adversarial malware exploits the low-confidence regions of models and can be identified using epistemic uncertainty of ML approaches -- epistemic uncertainty in a machine learning-based malware detector is a result of a lack of similar training samples in regions of the problem space. In particular, a Bayesian formulation can capture the model parameters' distribution and quantify epistemic uncertainty without sacrificing model performance. To verify our hypothesis, we consider Bayesian learning approaches with a mutual information-based formulation to quantify uncertainty and detect adversarial malware in Android, Windows domains and PDF malware. We found, quantifying uncertainty through Bayesian learning methods can defend against adversarial malware. In particular, Bayesian models: (1) are generally capable of identifying adversarial malware in both feature and problem space, (2) can detect concept drift by measuring uncertainty, and (3) with a diversity-promoting approach (or better posterior approximations) lead to parameter instances from the posterior to significantly enhance a detectors' ability.



## **26. Bidirectional Consistency Models**

cs.LG

40 pages, 25 figures

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.18035v1) [paper-pdf](http://arxiv.org/pdf/2403.18035v1)

**Authors**: Liangchen Li, Jiajun He

**Abstract**: Diffusion models (DMs) are capable of generating remarkably high-quality samples by iteratively denoising a random vector, a process that corresponds to moving along the probability flow ordinary differential equation (PF ODE). Interestingly, DMs can also invert an input image to noise by moving backward along the PF ODE, a key operation for downstream tasks such as interpolation and image editing. However, the iterative nature of this process restricts its speed, hindering its broader application. Recently, Consistency Models (CMs) have emerged to address this challenge by approximating the integral of the PF ODE, thereby bypassing the need to iterate. Yet, the absence of an explicit ODE solver complicates the inversion process. To resolve this, we introduce the Bidirectional Consistency Model (BCM), which learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation and inversion tasks within one framework. Notably, our proposed method enables one-step generation and inversion while also allowing the use of additional steps to enhance generation quality or reduce reconstruction error. Furthermore, by leveraging our model's bidirectional consistency, we introduce a sampling strategy that can enhance FID while preserving the generated image content. We further showcase our model's capabilities in several downstream tasks, such as interpolation and inpainting, and present demonstrations of potential applications, including blind restoration of compressed images and defending black-box adversarial attacks.



## **27. Analyzing the Quality Attributes of AI Vision Models in Open Repositories Under Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2401.12261v2) [paper-pdf](http://arxiv.org/pdf/2401.12261v2)

**Authors**: Zerui Wang, Yan Liu

**Abstract**: As AI models rapidly evolve, they are frequently released to open repositories, such as HuggingFace. It is essential to perform quality assurance validation on these models before integrating them into the production development lifecycle. In addition to evaluating efficiency in terms of balanced accuracy and computing costs, adversarial attacks are potential threats to the robustness and explainability of AI models. Meanwhile, XAI applies algorithms that approximate inputs to outputs post-hoc to identify the contributing features. Adversarial perturbations may also degrade the utility of XAI explanations that require further investigation. In this paper, we present an integrated process designed for downstream evaluation tasks, including validating AI model accuracy, evaluating robustness with benchmark perturbations, comparing explanation utility, and assessing overhead. We demonstrate an evaluation scenario involving six computer vision models, which include CNN-based, Transformer-based, and hybrid architectures, three types of perturbations, and five XAI methods, resulting in ninety unique combinations. The process reveals the explanation utility among the XAI methods in terms of the identified key areas responding to the adversarial perturbation. The process produces aggregated results that illustrate multiple attributes of each AI model.



## **28. Secure Aggregation is Not Private Against Membership Inference Attacks**

cs.LG

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17775v1) [paper-pdf](http://arxiv.org/pdf/2403.17775v1)

**Authors**: Khac-Hoang Ngo, Johan Östman, Giuseppe Durisi, Alexandre Graell i Amat

**Abstract**: Secure aggregation (SecAgg) is a commonly-used privacy-enhancing mechanism in federated learning, affording the server access only to the aggregate of model updates while safeguarding the confidentiality of individual updates. Despite widespread claims regarding SecAgg's privacy-preserving capabilities, a formal analysis of its privacy is lacking, making such presumptions unjustified. In this paper, we delve into the privacy implications of SecAgg by treating it as a local differential privacy (LDP) mechanism for each local update. We design a simple attack wherein an adversarial server seeks to discern which update vector a client submitted, out of two possible ones, in a single training round of federated learning under SecAgg. By conducting privacy auditing, we assess the success probability of this attack and quantify the LDP guarantees provided by SecAgg. Our numerical results unveil that, contrary to prevailing claims, SecAgg offers weak privacy against membership inference attacks even in a single training round. Indeed, it is difficult to hide a local update by adding other independent local updates when the updates are of high dimension. Our findings underscore the imperative for additional privacy-enhancing mechanisms, such as noise injection, in federated learning.



## **29. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17710v1) [paper-pdf](http://arxiv.org/pdf/2403.17710v1)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. Through extensive experiments, we showcase the capability of JudgeDeceiver in altering decision outcomes across various cases, highlighting the vulnerability of LLM-as-a-Judge systems to the optimization-based prompt injection attack.



## **30. Towards more Practical Threat Models in Artificial Intelligence Security**

cs.CR

18 pages, 4 figures, 8 tables, accepted to Usenix Security,  incorporated external feedback

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2311.09994v2) [paper-pdf](http://arxiv.org/pdf/2311.09994v2)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Alexandre Alahi

**Abstract**: Recent works have identified a gap between research and practice in artificial intelligence security: threats studied in academia do not always reflect the practical use and security risks of AI. For example, while models are often studied in isolation, they form part of larger ML pipelines in practice. Recent works also brought forward that adversarial manipulations introduced by academic attacks are impractical. We take a first step towards describing the full extent of this disparity. To this end, we revisit the threat models of the six most studied attacks in AI security research and match them to AI usage in practice via a survey with 271 industrial practitioners. On the one hand, we find that all existing threat models are indeed applicable. On the other hand, there are significant mismatches: research is often too generous with the attacker, assuming access to information not frequently available in real-world settings. Our paper is thus a call for action to study more practical threat models in artificial intelligence security.



## **31. Targeted Visualization of the Backbone of Encoder LLMs**

cs.LG

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.18872v1) [paper-pdf](http://arxiv.org/pdf/2403.18872v1)

**Authors**: Isaac Roberts, Alexander Schulz, Luca Hermes, Barbara Hammer

**Abstract**: Attention based Large Language Models (LLMs) are the state-of-the-art in natural language processing (NLP). The two most common architectures are encoders such as BERT, and decoders like the GPT models. Despite the success of encoder models, on which we focus in this work, they also bear several risks, including issues with bias or their susceptibility for adversarial attacks, signifying the necessity for explainable AI to detect such issues. While there does exist various local explainability methods focusing on the prediction of single inputs, global methods based on dimensionality reduction for classification inspection, which have emerged in other domains and that go further than just using t-SNE in the embedding space, are not widely spread in NLP.   To reduce this gap, we investigate the application of DeepView, a method for visualizing a part of the decision function together with a data set in two dimensions, to the NLP domain. While in previous work, DeepView has been used to inspect deep image classification models, we demonstrate how to apply it to BERT-based NLP classifiers and investigate its usability in this domain, including settings with adversarially perturbed input samples and pre-trained, fine-tuned, and multi-task models.



## **32. FaultGuard: A Generative Approach to Resilient Fault Prediction in Smart Electrical Grids**

cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17494v1) [paper-pdf](http://arxiv.org/pdf/2403.17494v1)

**Authors**: Emad Efatinasab, Francesco Marchiori, Alessandro Brighente, Mirco Rampazzo, Mauro Conti

**Abstract**: Predicting and classifying faults in electricity networks is crucial for uninterrupted provision and keeping maintenance costs at a minimum. Thanks to the advancements in the field provided by the smart grid, several data-driven approaches have been proposed in the literature to tackle fault prediction tasks. Implementing these systems brought several improvements, such as optimal energy consumption and quick restoration. Thus, they have become an essential component of the smart grid. However, the robustness and security of these systems against adversarial attacks have not yet been extensively investigated. These attacks can impair the whole grid and cause additional damage to the infrastructure, deceiving fault detection systems and disrupting restoration. In this paper, we present FaultGuard, the first framework for fault type and zone classification resilient to adversarial attacks. To ensure the security of our system, we employ an Anomaly Detection System (ADS) leveraging a novel Generative Adversarial Network training layer to identify attacks. Furthermore, we propose a low-complexity fault prediction model and an online adversarial training technique to enhance robustness. We comprehensively evaluate the framework's performance against various adversarial attacks using the IEEE13-AdvAttack dataset, which constitutes the state-of-the-art for resilient fault prediction benchmarking. Our model outclasses the state-of-the-art even without considering adversaries, with an accuracy of up to 0.958. Furthermore, our ADS shows attack detection capabilities with an accuracy of up to 1.000. Finally, we demonstrate how our novel training layers drastically increase performances across the whole framework, with a mean increase of 154% in ADS accuracy and 118% in model accuracy.



## **33. Rumor Detection with a novel graph neural network approach**

cs.AI

10 pages, 5 figures

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.16206v2) [paper-pdf](http://arxiv.org/pdf/2403.16206v2)

**Authors**: Tianrui Liu, Qi Cai, Changxin Xu, Bo Hong, Fanghao Ni, Yuxin Qiao, Tsungwei Yang

**Abstract**: The wide spread of rumors on social media has caused a negative impact on people's daily life, leading to potential panic, fear, and mental health problems for the public. How to debunk rumors as early as possible remains a challenging problem. Existing studies mainly leverage information propagation structure to detect rumors, while very few works focus on correlation among users that they may coordinate to spread rumors in order to gain large popularity. In this paper, we propose a new detection model, that jointly learns both the representations of user correlation and information propagation to detect rumors on social media. Specifically, we leverage graph neural networks to learn the representations of user correlation from a bipartite graph that describes the correlations between users and source tweets, and the representations of information propagation with a tree structure. Then we combine the learned representations from these two modules to classify the rumors. Since malicious users intend to subvert our model after deployment, we further develop a greedy attack scheme to analyze the cost of three adversarial attacks: graph attack, comment attack, and joint attack. Evaluation results on two public datasets illustrate that the proposed MODEL outperforms the state-of-the-art rumor detection models. We also demonstrate our method performs well for early rumor detection. Moreover, the proposed detection method is more robust to adversarial attacks compared to the best existing method. Importantly, we show that it requires a high cost for attackers to subvert user correlation pattern, demonstrating the importance of considering user correlation for rumor detection.



## **34. The Anatomy of Adversarial Attacks: Concept-based XAI Dissection**

cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16782v1) [paper-pdf](http://arxiv.org/pdf/2403.16782v1)

**Authors**: Georgii Mikriukov, Gesina Schwalbe, Franz Motzkus, Korinna Bade

**Abstract**: Adversarial attacks (AAs) pose a significant threat to the reliability and robustness of deep neural networks. While the impact of these attacks on model predictions has been extensively studied, their effect on the learned representations and concepts within these models remains largely unexplored. In this work, we perform an in-depth analysis of the influence of AAs on the concepts learned by convolutional neural networks (CNNs) using eXplainable artificial intelligence (XAI) techniques. Through an extensive set of experiments across various network architectures and targeted AA techniques, we unveil several key findings. First, AAs induce substantial alterations in the concept composition within the feature space, introducing new concepts or modifying existing ones. Second, the adversarial perturbation itself can be linearly decomposed into a set of latent vector components, with a subset of these being responsible for the attack's success. Notably, we discover that these components are target-specific, i.e., are similar for a given target class throughout different AA techniques and starting classes. Our findings provide valuable insights into the nature of AAs and their impact on learned representations, paving the way for the development of more robust and interpretable deep learning models, as well as effective defenses against adversarial threats.



## **35. DeepKnowledge: Generalisation-Driven Deep Learning Testing**

cs.LG

10 pages

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16768v1) [paper-pdf](http://arxiv.org/pdf/2403.16768v1)

**Authors**: Sondess Missaoui, Simos Gerasimou, Nikolaos Matragkas

**Abstract**: Despite their unprecedented success, DNNs are notoriously fragile to small shifts in data distribution, demanding effective testing techniques that can assess their dependability. Despite recent advances in DNN testing, there is a lack of systematic testing approaches that assess the DNN's capability to generalise and operate comparably beyond data in their training distribution. We address this gap with DeepKnowledge, a systematic testing methodology for DNN-based systems founded on the theory of knowledge generalisation, which aims to enhance DNN robustness and reduce the residual risk of 'black box' models. Conforming to this theory, DeepKnowledge posits that core computational DNN units, termed Transfer Knowledge neurons, can generalise under domain shift. DeepKnowledge provides an objective confidence measurement on testing activities of DNN given data distribution shifts and uses this information to instrument a generalisation-informed test adequacy criterion to check the transfer knowledge capacity of a test set. Our empirical evaluation of several DNNs, across multiple datasets and state-of-the-art adversarial generation techniques demonstrates the usefulness and effectiveness of DeepKnowledge and its ability to support the engineering of more dependable DNNs. We report improvements of up to 10 percentage points over state-of-the-art coverage criteria for detecting adversarial attacks on several benchmarks, including MNIST, SVHN, and CIFAR.



## **36. Boosting Adversarial Transferability by Block Shuffle and Rotation**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2308.10299v3) [paper-pdf](http://arxiv.org/pdf/2308.10299v3)

**Authors**: Kunyu Wang, Xuanran He, Wenxuan Wang, Xiaosen Wang

**Abstract**: Adversarial examples mislead deep neural networks with imperceptible perturbations and have brought significant threats to deep learning. An important aspect is their transferability, which refers to their ability to deceive other models, thus enabling attacks in the black-box setting. Though various methods have been proposed to boost transferability, the performance still falls short compared with white-box attacks. In this work, we observe that existing input transformation based attacks, one of the mainstream transfer-based attacks, result in different attention heatmaps on various models, which might limit the transferability. We also find that breaking the intrinsic relation of the image can disrupt the attention heatmap of the original image. Based on this finding, we propose a novel input transformation based attack called block shuffle and rotation (BSR). Specifically, BSR splits the input image into several blocks, then randomly shuffles and rotates these blocks to construct a set of new images for gradient calculation. Empirical evaluations on the ImageNet dataset demonstrate that BSR could achieve significantly better transferability than the existing input transformation based methods under single-model and ensemble-model settings. Combining BSR with the current input transformation method can further improve the transferability, which significantly outperforms the state-of-the-art methods. Code is available at https://github.com/Trustworthy-AI-Group/BSR



## **37. A Huber Loss Minimization Approach to Byzantine Robust Federated Learning**

cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2308.12581v2) [paper-pdf](http://arxiv.org/pdf/2308.12581v2)

**Authors**: Puning Zhao, Fei Yu, Zhiguo Wan

**Abstract**: Federated learning systems are susceptible to adversarial attacks. To combat this, we introduce a novel aggregator based on Huber loss minimization, and provide a comprehensive theoretical analysis. Under independent and identically distributed (i.i.d) assumption, our approach has several advantages compared to existing methods. Firstly, it has optimal dependence on $\epsilon$, which stands for the ratio of attacked clients. Secondly, our approach does not need precise knowledge of $\epsilon$. Thirdly, it allows different clients to have unequal data sizes. We then broaden our analysis to include non-i.i.d data, such that clients have slightly different distributions.



## **38. Revealing Vulnerabilities of Neural Networks in Parameter Learning and Defense Against Explanation-Aware Backdoors**

cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16569v1) [paper-pdf](http://arxiv.org/pdf/2403.16569v1)

**Authors**: Md Abdul Kadir, GowthamKrishna Addluri, Daniel Sonntag

**Abstract**: Explainable Artificial Intelligence (XAI) strategies play a crucial part in increasing the understanding and trustworthiness of neural networks. Nonetheless, these techniques could potentially generate misleading explanations. Blinding attacks can drastically alter a machine learning algorithm's prediction and explanation, providing misleading information by adding visually unnoticeable artifacts into the input, while maintaining the model's accuracy. It poses a serious challenge in ensuring the reliability of XAI methods. To ensure the reliability of XAI methods poses a real challenge, we leverage statistical analysis to highlight the changes in CNN weights within a CNN following blinding attacks. We introduce a method specifically designed to limit the effectiveness of such attacks during the evaluation phase, avoiding the need for extra training. The method we suggest defences against most modern explanation-aware adversarial attacks, achieving an approximate decrease of ~99\% in the Attack Success Rate (ASR) and a ~91\% reduction in the Mean Square Error (MSE) between the original explanation and the defended (post-attack) explanation across three unique types of attacks.



## **39. Exploring the Adversarial Capabilities of Large Language Models**

cs.AI

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2402.09132v3) [paper-pdf](http://arxiv.org/pdf/2402.09132v3)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.



## **40. On the resilience of Collaborative Learning-based Recommender Systems Against Community Detection Attack**

cs.IR

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2306.08929v2) [paper-pdf](http://arxiv.org/pdf/2306.08929v2)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning-based recommender systems emerged following the success of collaborative learning techniques such as Federated Learning (FL) and Gossip Learning (GL). In these systems, users participate in the training of a recommender system while maintaining their history of consumed items on their devices. While these solutions seemed appealing for preserving the privacy of the participants at first glance, recent studies have revealed that collaborative learning can be vulnerable to various privacy attacks. In this paper, we study the resilience of collaborative learning-based recommender systems against a novel privacy attack called Community Detection Attack (CDA). This attack enables an adversary to identify community members based on a chosen set of items (eg., identifying users interested in specific points-of-interest). Through experiments on three real recommendation datasets using two state-of-the-art recommendation models, we evaluate the sensitivity of an FL-based recommender system as well as two flavors of Gossip Learning-based recommender systems to CDA. The results show that across all models and datasets, the FL setting is more vulnerable to CDA compared to Gossip settings. Furthermore, we assess two off-the-shelf mitigation strategies, namely differential privacy (DP) and a \emph{Share less} policy, which consists of sharing a subset of less sensitive model parameters. The findings indicate a more favorable privacy-utility trade-off for the \emph{Share less} strategy, particularly in FedRecs.



## **41. Model-less Is the Best Model: Generating Pure Code Implementations to Replace On-Device DL Models**

cs.SE

Accepted by the ACM SIGSOFT International Symposium on Software  Testing and Analysis (ISSTA2024)

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16479v1) [paper-pdf](http://arxiv.org/pdf/2403.16479v1)

**Authors**: Mingyi Zhou, Xiang Gao, Pei Liu, John Grundy, Chunyang Chen, Xiao Chen, Li Li

**Abstract**: Recent studies show that deployed deep learning (DL) models such as those of Tensor Flow Lite (TFLite) can be easily extracted from real-world applications and devices by attackers to generate many kinds of attacks like adversarial attacks. Although securing deployed on-device DL models has gained increasing attention, no existing methods can fully prevent the aforementioned threats. Traditional software protection techniques have been widely explored, if on-device models can be implemented using pure code, such as C++, it will open the possibility of reusing existing software protection techniques. However, due to the complexity of DL models, there is no automatic method that can translate the DL models to pure code. To fill this gap, we propose a novel method, CustomDLCoder, to automatically extract the on-device model information and synthesize a customized executable program for a wide range of DL models. CustomDLCoder first parses the DL model, extracts its backend computing units, configures the computing units to a graph, and then generates customized code to implement and deploy the ML solution without explicit model representation. The synthesized program hides model information for DL deployment environments since it does not need to retain explicit model representation, preventing many attacks on the DL model. In addition, it improves ML performance because the customized code removes model parsing and preprocessing steps and only retains the data computing process. Our experimental results show that CustomDLCoder improves model security by disabling on-device model sniffing. Compared with the original on-device platform (i.e., TFLite), our method can accelerate model inference by 21.0% and 24.3% on x86-64 and ARM64 platforms, respectively. Most importantly, it can significantly reduce memory consumption by 68.8% and 36.0% on x86-64 and ARM64 platforms, respectively.



## **42. Secure Control of Connected and Automated Vehicles Using Trust-Aware Robust Event-Triggered Control Barrier Functions**

eess.SY

arXiv admin note: substantial text overlap with arXiv:2305.16818

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2401.02306v3) [paper-pdf](http://arxiv.org/pdf/2401.02306v3)

**Authors**: H M Sabbir Ahmad, Ehsan Sabouni, Akua Dickson, Wei Xiao, Christos G. Cassandras, Wenchao Li

**Abstract**: We address the security of a network of Connected and Automated Vehicles (CAVs) cooperating to safely navigate through a conflict area (e.g., traffic intersections, merging roadways, roundabouts). Previous studies have shown that such a network can be targeted by adversarial attacks causing traffic jams or safety violations ending in collisions. We focus on attacks targeting the V2X communication network used to share vehicle data and consider as well uncertainties due to noise in sensor measurements and communication channels. To combat these, motivated by recent work on the safe control of CAVs, we propose a trust-aware robust event-triggered decentralized control and coordination framework that can provably guarantee safety. We maintain a trust metric for each vehicle in the network computed based on their behavior and used to balance the tradeoff between conservativeness (when deeming every vehicle as untrustworthy) and guaranteed safety and security. It is important to highlight that our framework is invariant to the specific choice of the trust framework. Based on this framework, we propose an attack detection and mitigation scheme which has twofold benefits: (i) the trust framework is immune to false positives, and (ii) it provably guarantees safety against false positive cases. We use extensive simulations (in SUMO and CARLA) to validate the theoretical guarantees and demonstrate the efficacy of our proposed scheme to detect and mitigate adversarial attacks.



## **43. Ensemble Adversarial Defense via Integration of Multiple Dispersed Low Curvature Models**

cs.LG

Accepted to The 2024 International Joint Conference on Neural  Networks (IJCNN)

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16405v1) [paper-pdf](http://arxiv.org/pdf/2403.16405v1)

**Authors**: Kaikang Zhao, Xi Chen, Wei Huang, Liuxin Ding, Xianglong Kong, Fan Zhang

**Abstract**: The integration of an ensemble of deep learning models has been extensively explored to enhance defense against adversarial attacks. The diversity among sub-models increases the attack cost required to deceive the majority of the ensemble, thereby improving the adversarial robustness. While existing approaches mainly center on increasing diversity in feature representations or dispersion of first-order gradients with respect to input, the limited correlation between these diversity metrics and adversarial robustness constrains the performance of ensemble adversarial defense. In this work, we aim to enhance ensemble diversity by reducing attack transferability. We identify second-order gradients, which depict the loss curvature, as a key factor in adversarial robustness. Computing the Hessian matrix involved in second-order gradients is computationally expensive. To address this, we approximate the Hessian-vector product using differential approximation. Given that low curvature provides better robustness, our ensemble model was designed to consider the influence of curvature among different sub-models. We introduce a novel regularizer to train multiple more-diverse low-curvature network models. Extensive experiments across various datasets demonstrate that our ensemble model exhibits superior robustness against a range of attacks, underscoring the effectiveness of our approach.



## **44. Generating Potent Poisons and Backdoors from Scratch with Guided Diffusion**

cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16365v1) [paper-pdf](http://arxiv.org/pdf/2403.16365v1)

**Authors**: Hossein Souri, Arpit Bansal, Hamid Kazemi, Liam Fowl, Aniruddha Saha, Jonas Geiping, Andrew Gordon Wilson, Rama Chellappa, Tom Goldstein, Micah Goldblum

**Abstract**: Modern neural networks are often trained on massive datasets that are web scraped with minimal human inspection. As a result of this insecure curation pipeline, an adversary can poison or backdoor the resulting model by uploading malicious data to the internet and waiting for a victim to scrape and train on it. Existing approaches for creating poisons and backdoors start with randomly sampled clean data, called base samples, and then modify those samples to craft poisons. However, some base samples may be significantly more amenable to poisoning than others. As a result, we may be able to craft more potent poisons by carefully choosing the base samples. In this work, we use guided diffusion to synthesize base samples from scratch that lead to significantly more potent poisons and backdoors than previous state-of-the-art attacks. Our Guided Diffusion Poisoning (GDP) base samples can be combined with any downstream poisoning or backdoor attack to boost its effectiveness. Our implementation code is publicly available at: https://github.com/hsouri/GDP .



## **45. Subspace Defense: Discarding Adversarial Perturbations by Learning a Subspace for Clean Signals**

cs.LG

Accepted by COLING 2024

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.16176v1) [paper-pdf](http://arxiv.org/pdf/2403.16176v1)

**Authors**: Rui Zheng, Yuhao Zhou, Zhiheng Xi, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Deep neural networks (DNNs) are notoriously vulnerable to adversarial attacks that place carefully crafted perturbations on normal examples to fool DNNs. To better understand such attacks, a characterization of the features carried by adversarial examples is needed. In this paper, we tackle this challenge by inspecting the subspaces of sample features through spectral analysis. We first empirically show that the features of either clean signals or adversarial perturbations are redundant and span in low-dimensional linear subspaces respectively with minimal overlap, and the classical low-dimensional subspace projection can suppress perturbation features out of the subspace of clean signals. This makes it possible for DNNs to learn a subspace where only features of clean signals exist while those of perturbations are discarded, which can facilitate the distinction of adversarial examples. To prevent the residual perturbations that is inevitable in subspace learning, we propose an independence criterion to disentangle clean signals from perturbations. Experimental results show that the proposed strategy enables the model to inherently suppress adversaries, which not only boosts model robustness but also motivates new directions of effective adversarial defense.



## **46. Large Language Models for Blockchain Security: A Systematic Literature Review**

cs.CR

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.14280v2) [paper-pdf](http://arxiv.org/pdf/2403.14280v2)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.



## **47. ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations**

cs.LG

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2308.10457v5) [paper-pdf](http://arxiv.org/pdf/2308.10457v5)

**Authors**: Xinpeng Ling, Jie Fu, Kuncan Wang, Haitao Liu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks.   We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication rounds are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DPSGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of Differentially Private Federated Learning with Adaptive Local Iterations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and Cifar10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. Code is available at https://github.com/KnightWan/ALI-DPFL.



## **48. Robust Diffusion Models for Adversarial Purification**

cs.CV

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.16067v1) [paper-pdf](http://arxiv.org/pdf/2403.16067v1)

**Authors**: Guang Lin, Zerui Tao, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also mitigate the accuracy-robustness trade-off of DMs for the first time, which also provides DM-based AP an efficient adaptive ability to new attacks. Extensive experiments are conducted to demonstrate that our method achieves the state-of-the-art results and exhibits generalization against different attacks.



## **49. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

cs.CV

Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2310.11868v2) [paper-pdf](http://arxiv.org/pdf/2310.11868v2)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models.Through extensive benchmarking, we evaluate the robustness of five widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safety-driven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: This paper contains model outputs that may be offensive in nature.



## **50. An Embarrassingly Simple Defense Against Backdoor Attacks On SSL**

cs.CV

10 pages, 5 figures

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.15918v1) [paper-pdf](http://arxiv.org/pdf/2403.15918v1)

**Authors**: Aryan Satpathy, Nilaksh, Dhruva Rajwade

**Abstract**: Self Supervised Learning (SSL) has emerged as a powerful paradigm to tackle data landscapes with absence of human supervision. The ability to learn meaningful tasks without the use of labeled data makes SSL a popular method to manage large chunks of data in the absence of labels. However, recent work indicates SSL to be vulnerable to backdoor attacks, wherein models can be controlled, possibly maliciously, to suit an adversary's motives. Li et.al (2022) introduce a novel frequency-based backdoor attack: CTRL. They show that CTRL can be used to efficiently and stealthily gain control over a victim's model trained using SSL. In this work, we devise two defense strategies against frequency-based attacks in SSL: One applicable before model training and the second to be applied during model inference. Our first contribution utilizes the invariance property of the downstream task to defend against backdoor attacks in a generalizable fashion. We observe the ASR (Attack Success Rate) to reduce by over 60% across experiments. Our Inference-time defense relies on evasiveness of the attack and uses the luminance channel to defend against attacks. Using object classification as the downstream task for SSL, we demonstrate successful defense strategies that do not require re-training of the model. Code is available at https://github.com/Aryan-Satpathy/Backdoor.



