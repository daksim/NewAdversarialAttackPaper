# Latest Adversarial Attack Papers
**update at 2024-03-28 11:36:33**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Deep Learning for Robust and Explainable Models in Computer Vision**

cs.CV

150 pages, 37 figures, 12 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18674v1) [paper-pdf](http://arxiv.org/pdf/2403.18674v1)

**Authors**: Mohammadreza Amirian

**Abstract**: Recent breakthroughs in machine and deep learning (ML and DL) research have provided excellent tools for leveraging enormous amounts of data and optimizing huge models with millions of parameters to obtain accurate networks for image processing. These developments open up tremendous opportunities for using artificial intelligence (AI) in the automation and human assisted AI industry. However, as more and more models are deployed and used in practice, many challenges have emerged. This thesis presents various approaches that address robustness and explainability challenges for using ML and DL in practice.   Robustness and reliability are the critical components of any model before certification and deployment in practice. Deep convolutional neural networks (CNNs) exhibit vulnerability to transformations of their inputs, such as rotation and scaling, or intentional manipulations as described in the adversarial attack literature. In addition, building trust in AI-based models requires a better understanding of current models and developing methods that are more explainable and interpretable a priori.   This thesis presents developments in computer vision models' robustness and explainability. Furthermore, this thesis offers an example of using vision models' feature response visualization (models' interpretations) to improve robustness despite interpretability and robustness being seemingly unrelated in the related research. Besides methodological developments for robust and explainable vision models, a key message of this thesis is introducing model interpretation techniques as a tool for understanding vision models and improving their design and robustness. In addition to the theoretical developments, this thesis demonstrates several applications of ML and DL in different contexts, such as medical imaging and affective computing.



## **2. LCANets++: Robust Audio Classification using Multi-layer Neural Networks with Lateral Competition**

cs.SD

Accepted at 2024 IEEE International Conference on Acoustics, Speech  and Signal Processing Workshops (ICASSPW)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2308.12882v2) [paper-pdf](http://arxiv.org/pdf/2308.12882v2)

**Authors**: Sayanton V. Dibbo, Juston S. Moore, Garrett T. Kenyon, Michael A. Teti

**Abstract**: Audio classification aims at recognizing audio signals, including speech commands or sound events. However, current audio classifiers are susceptible to perturbations and adversarial attacks. In addition, real-world audio classification tasks often suffer from limited labeled data. To help bridge these gaps, previous work developed neuro-inspired convolutional neural networks (CNNs) with sparse coding via the Locally Competitive Algorithm (LCA) in the first layer (i.e., LCANets) for computer vision. LCANets learn in a combination of supervised and unsupervised learning, reducing dependency on labeled samples. Motivated by the fact that auditory cortex is also sparse, we extend LCANets to audio recognition tasks and introduce LCANets++, which are CNNs that perform sparse coding in multiple layers via LCA. We demonstrate that LCANets++ are more robust than standard CNNs and LCANets against perturbations, e.g., background noise, as well as black-box and white-box attacks, e.g., evasion and fast gradient sign (FGSM) attacks.



## **3. The Impact of Uniform Inputs on Activation Sparsity and Energy-Latency Attacks in Computer Vision**

cs.CR

Accepted at the DLSP 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18587v1) [paper-pdf](http://arxiv.org/pdf/2403.18587v1)

**Authors**: Andreas Müller, Erwin Quiring

**Abstract**: Resource efficiency plays an important role for machine learning nowadays. The energy and decision latency are two critical aspects to ensure a sustainable and practical application. Unfortunately, the energy consumption and decision latency are not robust against adversaries. Researchers have recently demonstrated that attackers can compute and submit so-called sponge examples at inference time to increase the energy consumption and decision latency of neural networks. In computer vision, the proposed strategy crafts inputs with less activation sparsity which could otherwise be used to accelerate the computation. In this paper, we analyze the mechanism how these energy-latency attacks reduce activation sparsity. In particular, we find that input uniformity is a key enabler. A uniform image, that is, an image with mostly flat, uniformly colored surfaces, triggers more activations due to a specific interplay of convolution, batch normalization, and ReLU activation. Based on these insights, we propose two new simple, yet effective strategies for crafting sponge examples: sampling images from a probability distribution and identifying dense, yet inconspicuous inputs in natural datasets. We empirically examine our findings in a comprehensive evaluation with multiple image classification models and show that our attack achieves the same sparsity effect as prior sponge-example methods, but at a fraction of computation effort. We also show that our sponge examples transfer between different neural networks. Finally, we discuss applications of our findings for the good by improving efficiency by increasing sparsity.



## **4. MisGUIDE : Defense Against Data-Free Deep Learning Model Extraction**

cs.CR

Under Review

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18580v1) [paper-pdf](http://arxiv.org/pdf/2403.18580v1)

**Authors**: Mahendra Gurve, Sankar Behera, Satyadev Ahlawat, Yamuna Prasad

**Abstract**: The rise of Machine Learning as a Service (MLaaS) has led to the widespread deployment of machine learning models trained on diverse datasets. These models are employed for predictive services through APIs, raising concerns about the security and confidentiality of the models due to emerging vulnerabilities in prediction APIs. Of particular concern are model cloning attacks, where individuals with limited data and no knowledge of the training dataset manage to replicate a victim model's functionality through black-box query access. This commonly entails generating adversarial queries to query the victim model, thereby creating a labeled dataset.   This paper proposes "MisGUIDE", a two-step defense framework for Deep Learning models that disrupts the adversarial sample generation process by providing a probabilistic response when the query is deemed OOD. The first step employs a Vision Transformer-based framework to identify OOD queries, while the second step perturbs the response for such queries, introducing a probabilistic loss function to MisGUIDE the attackers. The aim of the proposed defense method is to reduce the accuracy of the cloned model while maintaining accuracy on authentic queries. Extensive experiments conducted on two benchmark datasets demonstrate that the proposed framework significantly enhances the resistance against state-of-the-art data-free model extraction in black-box settings.



## **5. CosalPure: Learning Concept from Group Images for Robust Co-Saliency Detection**

cs.CV

8 pages

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18554v1) [paper-pdf](http://arxiv.org/pdf/2403.18554v1)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstract**: Co-salient object detection (CoSOD) aims to identify the common and salient (usually in the foreground) regions across a given group of images. Although achieving significant progress, state-of-the-art CoSODs could be easily affected by some adversarial perturbations, leading to substantial accuracy reduction. The adversarial perturbations can mislead CoSODs but do not change the high-level semantic information (e.g., concept) of the co-salient objects. In this paper, we propose a novel robustness enhancement framework by first learning the concept of the co-salient objects based on the input group images and then leveraging this concept to purify adversarial perturbations, which are subsequently fed to CoSODs for robustness enhancement. Specifically, we propose CosalPure containing two modules, i.e., group-image concept learning and concept-guided diffusion purification. For the first module, we adopt a pre-trained text-to-image diffusion model to learn the concept of co-salient objects within group images where the learned concept is robust to adversarial examples. For the second module, we map the adversarial image to the latent space and then perform diffusion generation by embedding the learned concept into the noise prediction function as an extra condition. Our method can effectively alleviate the influence of the SOTA adversarial attack containing different adversarial patterns, including exposure and noise. The extensive results demonstrate that our method could enhance the robustness of CoSODs significantly.



## **6. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2311.08268v3) [paper-pdf](http://arxiv.org/pdf/2311.08268v3)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.



## **7. Attack and Defense Analysis of Learned Image Compression**

eess.IV

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2401.10345v3) [paper-pdf](http://arxiv.org/pdf/2401.10345v3)

**Authors**: Tianyu Zhu, Heming Sun, Xiankui Xiong, Xuanpeng Zhu, Yong Gong, Minge jing, Yibo Fan

**Abstract**: Learned image compression (LIC) is becoming more and more popular these years with its high efficiency and outstanding compression quality. Still, the practicality against modified inputs added with specific noise could not be ignored. White-box attacks such as FGSM and PGD use only gradient to compute adversarial images that mislead LIC models to output unexpected results. Our experiments compare the effects of different dimensions such as attack methods, models, qualities, and targets, concluding that in the worst case, there is a 61.55% decrease in PSNR or a 19.15 times increase in bpp under the PGD attack. To improve their robustness, we conduct adversarial training by adding adversarial images into the training datasets, which obtains a 95.52% decrease in the R-D cost of the most vulnerable LIC model. We further test the robustness of H.266, whose better performance on reconstruction quality extends its possibility to defend one-step or iterative adversarial attacks.



## **8. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.16432v2) [paper-pdf](http://arxiv.org/pdf/2403.16432v2)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo.



## **9. SemRoDe: Macro Adversarial Training to Learn Representations That are Robust to Word-Level Attacks**

cs.CL

Published in NAACL 2024 (Main Track)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18423v1) [paper-pdf](http://arxiv.org/pdf/2403.18423v1)

**Authors**: Brian Formento, Wenjie Feng, Chuan Sheng Foo, Luu Anh Tuan, See-Kiong Ng

**Abstract**: Language models (LMs) are indispensable tools for natural language processing tasks, but their vulnerability to adversarial attacks remains a concern. While current research has explored adversarial training techniques, their improvements to defend against word-level attacks have been limited. In this work, we propose a novel approach called Semantic Robust Defence (SemRoDe), a Macro Adversarial Training strategy to enhance the robustness of LMs. Drawing inspiration from recent studies in the image domain, we investigate and later confirm that in a discrete data setting such as language, adversarial samples generated via word substitutions do indeed belong to an adversarial domain exhibiting a high Wasserstein distance from the base domain. Our method learns a robust representation that bridges these two domains. We hypothesize that if samples were not projected into an adversarial domain, but instead to a domain with minimal shift, it would improve attack robustness. We align the domains by incorporating a new distance-based objective. With this, our model is able to learn more generalized representations by aligning the model's high-level output features and therefore better handling unseen adversarial samples. This method can be generalized across word embeddings, even when they share minimal overlap at both vocabulary and word-substitution levels. To evaluate the effectiveness of our approach, we conduct experiments on BERT and RoBERTa models on three datasets. The results demonstrate promising state-of-the-art robustness.



## **10. LocalStyleFool: Regional Video Style Transfer Attack Using Segment Anything Model**

cs.CV

Accepted to 2024 IEEE Security and Privacy Workshops (SPW)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.11656v2) [paper-pdf](http://arxiv.org/pdf/2403.11656v2)

**Authors**: Yuxin Cao, Jinghao Li, Xi Xiao, Derui Wang, Minhui Xue, Hao Ge, Wei Liu, Guangwu Hu

**Abstract**: Previous work has shown that well-crafted adversarial perturbations can threaten the security of video recognition systems. Attackers can invade such models with a low query budget when the perturbations are semantic-invariant, such as StyleFool. Despite the query efficiency, the naturalness of the minutia areas still requires amelioration, since StyleFool leverages style transfer to all pixels in each frame. To close the gap, we propose LocalStyleFool, an improved black-box video adversarial attack that superimposes regional style-transfer-based perturbations on videos. Benefiting from the popularity and scalably usability of Segment Anything Model (SAM), we first extract different regions according to semantic information and then track them through the video stream to maintain the temporal consistency. Then, we add style-transfer-based perturbations to several regions selected based on the associative criterion of transfer-based gradient information and regional area. Perturbation fine adjustment is followed to make stylized videos adversarial. We demonstrate that LocalStyleFool can improve both intra-frame and inter-frame naturalness through a human-assessed survey, while maintaining competitive fooling rate and query efficiency. Successful experiments on the high-resolution dataset also showcase that scrupulous segmentation of SAM helps to improve the scalability of adversarial attacks under high-resolution data.



## **11. Physical 3D Adversarial Attacks against Monocular Depth Estimation in Autonomous Driving**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.17301v2) [paper-pdf](http://arxiv.org/pdf/2403.17301v2)

**Authors**: Junhao Zheng, Chenhao Lin, Jiahao Sun, Zhengyu Zhao, Qian Li, Chao Shen

**Abstract**: Deep learning-based monocular depth estimation (MDE), extensively applied in autonomous driving, is known to be vulnerable to adversarial attacks. Previous physical attacks against MDE models rely on 2D adversarial patches, so they only affect a small, localized region in the MDE map but fail under various viewpoints. To address these limitations, we propose 3D Depth Fool (3D$^2$Fool), the first 3D texture-based adversarial attack against MDE models. 3D$^2$Fool is specifically optimized to generate 3D adversarial textures agnostic to model types of vehicles and to have improved robustness in bad weather conditions, such as rain and fog. Experimental results validate the superior performance of our 3D$^2$Fool across various scenarios, including vehicles, MDE models, weather conditions, and viewpoints. Real-world experiments with printed 3D textures on physical vehicle models further demonstrate that our 3D$^2$Fool can cause an MDE error of over 10 meters.



## **12. Uncertainty-Aware SAR ATR: Defending Against Adversarial Attacks via Bayesian Neural Networks**

cs.CV

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18318v1) [paper-pdf](http://arxiv.org/pdf/2403.18318v1)

**Authors**: Tian Ye, Rajgopal Kannan, Viktor Prasanna, Carl Busart

**Abstract**: Adversarial attacks have demonstrated the vulnerability of Machine Learning (ML) image classifiers in Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) systems. An adversarial attack can deceive the classifier into making incorrect predictions by perturbing the input SAR images, for example, with a few scatterers attached to the on-ground objects. Therefore, it is critical to develop robust SAR ATR systems that can detect potential adversarial attacks by leveraging the inherent uncertainty in ML classifiers, thereby effectively alerting human decision-makers. In this paper, we propose a novel uncertainty-aware SAR ATR for detecting adversarial attacks. Specifically, we leverage the capability of Bayesian Neural Networks (BNNs) in performing image classification with quantified epistemic uncertainty to measure the confidence for each input SAR image. By evaluating the uncertainty, our method alerts when the input SAR image is likely to be adversarially generated. Simultaneously, we also generate visual explanations that reveal the specific regions in the SAR image where the adversarial scatterers are likely to to be present, thus aiding human decision-making with hints of evidence of adversarial attacks. Experiments on the MSTAR dataset demonstrate that our approach can identify over 80% adversarial SAR images with fewer than 20% false alarms, and our visual explanations can identify up to over 90% of scatterers in an adversarial SAR image.



## **13. Bayesian Learned Models Can Detect Adversarial Malware For Free**

cs.CR

Accepted to the 29th European Symposium on Research in Computer  Security (ESORICS) 2024 Conference

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18309v1) [paper-pdf](http://arxiv.org/pdf/2403.18309v1)

**Authors**: Bao Gia Doan, Dang Quang Nguyen, Paul Montague, Tamas Abraham, Olivier De Vel, Seyit Camtepe, Salil S. Kanhere, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: The vulnerability of machine learning-based malware detectors to adversarial attacks has prompted the need for robust solutions. Adversarial training is an effective method but is computationally expensive to scale up to large datasets and comes at the cost of sacrificing model performance for robustness. We hypothesize that adversarial malware exploits the low-confidence regions of models and can be identified using epistemic uncertainty of ML approaches -- epistemic uncertainty in a machine learning-based malware detector is a result of a lack of similar training samples in regions of the problem space. In particular, a Bayesian formulation can capture the model parameters' distribution and quantify epistemic uncertainty without sacrificing model performance. To verify our hypothesis, we consider Bayesian learning approaches with a mutual information-based formulation to quantify uncertainty and detect adversarial malware in Android, Windows domains and PDF malware. We found, quantifying uncertainty through Bayesian learning methods can defend against adversarial malware. In particular, Bayesian models: (1) are generally capable of identifying adversarial malware in both feature and problem space, (2) can detect concept drift by measuring uncertainty, and (3) with a diversity-promoting approach (or better posterior approximations) lead to parameter instances from the posterior to significantly enhance a detectors' ability.



## **14. Bidirectional Consistency Models**

cs.LG

40 pages, 25 figures

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.18035v1) [paper-pdf](http://arxiv.org/pdf/2403.18035v1)

**Authors**: Liangchen Li, Jiajun He

**Abstract**: Diffusion models (DMs) are capable of generating remarkably high-quality samples by iteratively denoising a random vector, a process that corresponds to moving along the probability flow ordinary differential equation (PF ODE). Interestingly, DMs can also invert an input image to noise by moving backward along the PF ODE, a key operation for downstream tasks such as interpolation and image editing. However, the iterative nature of this process restricts its speed, hindering its broader application. Recently, Consistency Models (CMs) have emerged to address this challenge by approximating the integral of the PF ODE, thereby bypassing the need to iterate. Yet, the absence of an explicit ODE solver complicates the inversion process. To resolve this, we introduce the Bidirectional Consistency Model (BCM), which learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation and inversion tasks within one framework. Notably, our proposed method enables one-step generation and inversion while also allowing the use of additional steps to enhance generation quality or reduce reconstruction error. Furthermore, by leveraging our model's bidirectional consistency, we introduce a sampling strategy that can enhance FID while preserving the generated image content. We further showcase our model's capabilities in several downstream tasks, such as interpolation and inpainting, and present demonstrations of potential applications, including blind restoration of compressed images and defending black-box adversarial attacks.



## **15. Analyzing the Quality Attributes of AI Vision Models in Open Repositories Under Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2401.12261v2) [paper-pdf](http://arxiv.org/pdf/2401.12261v2)

**Authors**: Zerui Wang, Yan Liu

**Abstract**: As AI models rapidly evolve, they are frequently released to open repositories, such as HuggingFace. It is essential to perform quality assurance validation on these models before integrating them into the production development lifecycle. In addition to evaluating efficiency in terms of balanced accuracy and computing costs, adversarial attacks are potential threats to the robustness and explainability of AI models. Meanwhile, XAI applies algorithms that approximate inputs to outputs post-hoc to identify the contributing features. Adversarial perturbations may also degrade the utility of XAI explanations that require further investigation. In this paper, we present an integrated process designed for downstream evaluation tasks, including validating AI model accuracy, evaluating robustness with benchmark perturbations, comparing explanation utility, and assessing overhead. We demonstrate an evaluation scenario involving six computer vision models, which include CNN-based, Transformer-based, and hybrid architectures, three types of perturbations, and five XAI methods, resulting in ninety unique combinations. The process reveals the explanation utility among the XAI methods in terms of the identified key areas responding to the adversarial perturbation. The process produces aggregated results that illustrate multiple attributes of each AI model.



## **16. Secure Aggregation is Not Private Against Membership Inference Attacks**

cs.LG

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17775v1) [paper-pdf](http://arxiv.org/pdf/2403.17775v1)

**Authors**: Khac-Hoang Ngo, Johan Östman, Giuseppe Durisi, Alexandre Graell i Amat

**Abstract**: Secure aggregation (SecAgg) is a commonly-used privacy-enhancing mechanism in federated learning, affording the server access only to the aggregate of model updates while safeguarding the confidentiality of individual updates. Despite widespread claims regarding SecAgg's privacy-preserving capabilities, a formal analysis of its privacy is lacking, making such presumptions unjustified. In this paper, we delve into the privacy implications of SecAgg by treating it as a local differential privacy (LDP) mechanism for each local update. We design a simple attack wherein an adversarial server seeks to discern which update vector a client submitted, out of two possible ones, in a single training round of federated learning under SecAgg. By conducting privacy auditing, we assess the success probability of this attack and quantify the LDP guarantees provided by SecAgg. Our numerical results unveil that, contrary to prevailing claims, SecAgg offers weak privacy against membership inference attacks even in a single training round. Indeed, it is difficult to hide a local update by adding other independent local updates when the updates are of high dimension. Our findings underscore the imperative for additional privacy-enhancing mechanisms, such as noise injection, in federated learning.



## **17. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17710v1) [paper-pdf](http://arxiv.org/pdf/2403.17710v1)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. Through extensive experiments, we showcase the capability of JudgeDeceiver in altering decision outcomes across various cases, highlighting the vulnerability of LLM-as-a-Judge systems to the optimization-based prompt injection attack.



## **18. Towards more Practical Threat Models in Artificial Intelligence Security**

cs.CR

18 pages, 4 figures, 8 tables, accepted to Usenix Security,  incorporated external feedback

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2311.09994v2) [paper-pdf](http://arxiv.org/pdf/2311.09994v2)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Alexandre Alahi

**Abstract**: Recent works have identified a gap between research and practice in artificial intelligence security: threats studied in academia do not always reflect the practical use and security risks of AI. For example, while models are often studied in isolation, they form part of larger ML pipelines in practice. Recent works also brought forward that adversarial manipulations introduced by academic attacks are impractical. We take a first step towards describing the full extent of this disparity. To this end, we revisit the threat models of the six most studied attacks in AI security research and match them to AI usage in practice via a survey with 271 industrial practitioners. On the one hand, we find that all existing threat models are indeed applicable. On the other hand, there are significant mismatches: research is often too generous with the attacker, assuming access to information not frequently available in real-world settings. Our paper is thus a call for action to study more practical threat models in artificial intelligence security.



## **19. FaultGuard: A Generative Approach to Resilient Fault Prediction in Smart Electrical Grids**

cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17494v1) [paper-pdf](http://arxiv.org/pdf/2403.17494v1)

**Authors**: Emad Efatinasab, Francesco Marchiori, Alessandro Brighente, Mirco Rampazzo, Mauro Conti

**Abstract**: Predicting and classifying faults in electricity networks is crucial for uninterrupted provision and keeping maintenance costs at a minimum. Thanks to the advancements in the field provided by the smart grid, several data-driven approaches have been proposed in the literature to tackle fault prediction tasks. Implementing these systems brought several improvements, such as optimal energy consumption and quick restoration. Thus, they have become an essential component of the smart grid. However, the robustness and security of these systems against adversarial attacks have not yet been extensively investigated. These attacks can impair the whole grid and cause additional damage to the infrastructure, deceiving fault detection systems and disrupting restoration. In this paper, we present FaultGuard, the first framework for fault type and zone classification resilient to adversarial attacks. To ensure the security of our system, we employ an Anomaly Detection System (ADS) leveraging a novel Generative Adversarial Network training layer to identify attacks. Furthermore, we propose a low-complexity fault prediction model and an online adversarial training technique to enhance robustness. We comprehensively evaluate the framework's performance against various adversarial attacks using the IEEE13-AdvAttack dataset, which constitutes the state-of-the-art for resilient fault prediction benchmarking. Our model outclasses the state-of-the-art even without considering adversaries, with an accuracy of up to 0.958. Furthermore, our ADS shows attack detection capabilities with an accuracy of up to 1.000. Finally, we demonstrate how our novel training layers drastically increase performances across the whole framework, with a mean increase of 154% in ADS accuracy and 118% in model accuracy.



## **20. Rumor Detection with a novel graph neural network approach**

cs.AI

10 pages, 5 figures

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.16206v2) [paper-pdf](http://arxiv.org/pdf/2403.16206v2)

**Authors**: Tianrui Liu, Qi Cai, Changxin Xu, Bo Hong, Fanghao Ni, Yuxin Qiao, Tsungwei Yang

**Abstract**: The wide spread of rumors on social media has caused a negative impact on people's daily life, leading to potential panic, fear, and mental health problems for the public. How to debunk rumors as early as possible remains a challenging problem. Existing studies mainly leverage information propagation structure to detect rumors, while very few works focus on correlation among users that they may coordinate to spread rumors in order to gain large popularity. In this paper, we propose a new detection model, that jointly learns both the representations of user correlation and information propagation to detect rumors on social media. Specifically, we leverage graph neural networks to learn the representations of user correlation from a bipartite graph that describes the correlations between users and source tweets, and the representations of information propagation with a tree structure. Then we combine the learned representations from these two modules to classify the rumors. Since malicious users intend to subvert our model after deployment, we further develop a greedy attack scheme to analyze the cost of three adversarial attacks: graph attack, comment attack, and joint attack. Evaluation results on two public datasets illustrate that the proposed MODEL outperforms the state-of-the-art rumor detection models. We also demonstrate our method performs well for early rumor detection. Moreover, the proposed detection method is more robust to adversarial attacks compared to the best existing method. Importantly, we show that it requires a high cost for attackers to subvert user correlation pattern, demonstrating the importance of considering user correlation for rumor detection.



## **21. The Anatomy of Adversarial Attacks: Concept-based XAI Dissection**

cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16782v1) [paper-pdf](http://arxiv.org/pdf/2403.16782v1)

**Authors**: Georgii Mikriukov, Gesina Schwalbe, Franz Motzkus, Korinna Bade

**Abstract**: Adversarial attacks (AAs) pose a significant threat to the reliability and robustness of deep neural networks. While the impact of these attacks on model predictions has been extensively studied, their effect on the learned representations and concepts within these models remains largely unexplored. In this work, we perform an in-depth analysis of the influence of AAs on the concepts learned by convolutional neural networks (CNNs) using eXplainable artificial intelligence (XAI) techniques. Through an extensive set of experiments across various network architectures and targeted AA techniques, we unveil several key findings. First, AAs induce substantial alterations in the concept composition within the feature space, introducing new concepts or modifying existing ones. Second, the adversarial perturbation itself can be linearly decomposed into a set of latent vector components, with a subset of these being responsible for the attack's success. Notably, we discover that these components are target-specific, i.e., are similar for a given target class throughout different AA techniques and starting classes. Our findings provide valuable insights into the nature of AAs and their impact on learned representations, paving the way for the development of more robust and interpretable deep learning models, as well as effective defenses against adversarial threats.



## **22. DeepKnowledge: Generalisation-Driven Deep Learning Testing**

cs.LG

10 pages

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16768v1) [paper-pdf](http://arxiv.org/pdf/2403.16768v1)

**Authors**: Sondess Missaoui, Simos Gerasimou, Nikolaos Matragkas

**Abstract**: Despite their unprecedented success, DNNs are notoriously fragile to small shifts in data distribution, demanding effective testing techniques that can assess their dependability. Despite recent advances in DNN testing, there is a lack of systematic testing approaches that assess the DNN's capability to generalise and operate comparably beyond data in their training distribution. We address this gap with DeepKnowledge, a systematic testing methodology for DNN-based systems founded on the theory of knowledge generalisation, which aims to enhance DNN robustness and reduce the residual risk of 'black box' models. Conforming to this theory, DeepKnowledge posits that core computational DNN units, termed Transfer Knowledge neurons, can generalise under domain shift. DeepKnowledge provides an objective confidence measurement on testing activities of DNN given data distribution shifts and uses this information to instrument a generalisation-informed test adequacy criterion to check the transfer knowledge capacity of a test set. Our empirical evaluation of several DNNs, across multiple datasets and state-of-the-art adversarial generation techniques demonstrates the usefulness and effectiveness of DeepKnowledge and its ability to support the engineering of more dependable DNNs. We report improvements of up to 10 percentage points over state-of-the-art coverage criteria for detecting adversarial attacks on several benchmarks, including MNIST, SVHN, and CIFAR.



## **23. Boosting Adversarial Transferability by Block Shuffle and Rotation**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2308.10299v3) [paper-pdf](http://arxiv.org/pdf/2308.10299v3)

**Authors**: Kunyu Wang, Xuanran He, Wenxuan Wang, Xiaosen Wang

**Abstract**: Adversarial examples mislead deep neural networks with imperceptible perturbations and have brought significant threats to deep learning. An important aspect is their transferability, which refers to their ability to deceive other models, thus enabling attacks in the black-box setting. Though various methods have been proposed to boost transferability, the performance still falls short compared with white-box attacks. In this work, we observe that existing input transformation based attacks, one of the mainstream transfer-based attacks, result in different attention heatmaps on various models, which might limit the transferability. We also find that breaking the intrinsic relation of the image can disrupt the attention heatmap of the original image. Based on this finding, we propose a novel input transformation based attack called block shuffle and rotation (BSR). Specifically, BSR splits the input image into several blocks, then randomly shuffles and rotates these blocks to construct a set of new images for gradient calculation. Empirical evaluations on the ImageNet dataset demonstrate that BSR could achieve significantly better transferability than the existing input transformation based methods under single-model and ensemble-model settings. Combining BSR with the current input transformation method can further improve the transferability, which significantly outperforms the state-of-the-art methods. Code is available at https://github.com/Trustworthy-AI-Group/BSR



## **24. A Huber Loss Minimization Approach to Byzantine Robust Federated Learning**

cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2308.12581v2) [paper-pdf](http://arxiv.org/pdf/2308.12581v2)

**Authors**: Puning Zhao, Fei Yu, Zhiguo Wan

**Abstract**: Federated learning systems are susceptible to adversarial attacks. To combat this, we introduce a novel aggregator based on Huber loss minimization, and provide a comprehensive theoretical analysis. Under independent and identically distributed (i.i.d) assumption, our approach has several advantages compared to existing methods. Firstly, it has optimal dependence on $\epsilon$, which stands for the ratio of attacked clients. Secondly, our approach does not need precise knowledge of $\epsilon$. Thirdly, it allows different clients to have unequal data sizes. We then broaden our analysis to include non-i.i.d data, such that clients have slightly different distributions.



## **25. Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy**

cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16591v1) [paper-pdf](http://arxiv.org/pdf/2403.16591v1)

**Authors**: Xiaojin Zhang, Yulin Fei, Wei Chen, Hai Jin

**Abstract**: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between local differential privacy and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. Our theoretical contributions are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Maximum Bayesian Privacy (MBP), encapsulated by equations $\epsilon_{p,a} \leq \frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p,m} + \epsilon)\cdot(e^{\epsilon_{p,m} + \epsilon} - 1)}$ and the equivalence between $\xi$-MBP and $2\xi$-LDP established under uniform prior distribution. These relationships fortify our understanding of the privacy guarantees provided by various mechanisms, leading to the realization that a mechanism satisfying $\xi$-LDP also confers $\xi$-MBP, and vice versa. Our work not only lays the groundwork for future empirical exploration but also promises to enhance the design of privacy-preserving algorithms that do not compromise on utility, thereby fostering the development of trustworthy machine learning solutions.



## **26. Revealing Vulnerabilities of Neural Networks in Parameter Learning and Defense Against Explanation-Aware Backdoors**

cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16569v1) [paper-pdf](http://arxiv.org/pdf/2403.16569v1)

**Authors**: Md Abdul Kadir, GowthamKrishna Addluri, Daniel Sonntag

**Abstract**: Explainable Artificial Intelligence (XAI) strategies play a crucial part in increasing the understanding and trustworthiness of neural networks. Nonetheless, these techniques could potentially generate misleading explanations. Blinding attacks can drastically alter a machine learning algorithm's prediction and explanation, providing misleading information by adding visually unnoticeable artifacts into the input, while maintaining the model's accuracy. It poses a serious challenge in ensuring the reliability of XAI methods. To ensure the reliability of XAI methods poses a real challenge, we leverage statistical analysis to highlight the changes in CNN weights within a CNN following blinding attacks. We introduce a method specifically designed to limit the effectiveness of such attacks during the evaluation phase, avoiding the need for extra training. The method we suggest defences against most modern explanation-aware adversarial attacks, achieving an approximate decrease of ~99\% in the Attack Success Rate (ASR) and a ~91\% reduction in the Mean Square Error (MSE) between the original explanation and the defended (post-attack) explanation across three unique types of attacks.



## **27. Exploring the Adversarial Capabilities of Large Language Models**

cs.AI

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2402.09132v3) [paper-pdf](http://arxiv.org/pdf/2402.09132v3)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.



## **28. On the resilience of Collaborative Learning-based Recommender Systems Against Community Detection Attack**

cs.IR

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2306.08929v2) [paper-pdf](http://arxiv.org/pdf/2306.08929v2)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning-based recommender systems emerged following the success of collaborative learning techniques such as Federated Learning (FL) and Gossip Learning (GL). In these systems, users participate in the training of a recommender system while maintaining their history of consumed items on their devices. While these solutions seemed appealing for preserving the privacy of the participants at first glance, recent studies have revealed that collaborative learning can be vulnerable to various privacy attacks. In this paper, we study the resilience of collaborative learning-based recommender systems against a novel privacy attack called Community Detection Attack (CDA). This attack enables an adversary to identify community members based on a chosen set of items (eg., identifying users interested in specific points-of-interest). Through experiments on three real recommendation datasets using two state-of-the-art recommendation models, we evaluate the sensitivity of an FL-based recommender system as well as two flavors of Gossip Learning-based recommender systems to CDA. The results show that across all models and datasets, the FL setting is more vulnerable to CDA compared to Gossip settings. Furthermore, we assess two off-the-shelf mitigation strategies, namely differential privacy (DP) and a \emph{Share less} policy, which consists of sharing a subset of less sensitive model parameters. The findings indicate a more favorable privacy-utility trade-off for the \emph{Share less} strategy, particularly in FedRecs.



## **29. Model-less Is the Best Model: Generating Pure Code Implementations to Replace On-Device DL Models**

cs.SE

Accepted by the ACM SIGSOFT International Symposium on Software  Testing and Analysis (ISSTA2024)

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16479v1) [paper-pdf](http://arxiv.org/pdf/2403.16479v1)

**Authors**: Mingyi Zhou, Xiang Gao, Pei Liu, John Grundy, Chunyang Chen, Xiao Chen, Li Li

**Abstract**: Recent studies show that deployed deep learning (DL) models such as those of Tensor Flow Lite (TFLite) can be easily extracted from real-world applications and devices by attackers to generate many kinds of attacks like adversarial attacks. Although securing deployed on-device DL models has gained increasing attention, no existing methods can fully prevent the aforementioned threats. Traditional software protection techniques have been widely explored, if on-device models can be implemented using pure code, such as C++, it will open the possibility of reusing existing software protection techniques. However, due to the complexity of DL models, there is no automatic method that can translate the DL models to pure code. To fill this gap, we propose a novel method, CustomDLCoder, to automatically extract the on-device model information and synthesize a customized executable program for a wide range of DL models. CustomDLCoder first parses the DL model, extracts its backend computing units, configures the computing units to a graph, and then generates customized code to implement and deploy the ML solution without explicit model representation. The synthesized program hides model information for DL deployment environments since it does not need to retain explicit model representation, preventing many attacks on the DL model. In addition, it improves ML performance because the customized code removes model parsing and preprocessing steps and only retains the data computing process. Our experimental results show that CustomDLCoder improves model security by disabling on-device model sniffing. Compared with the original on-device platform (i.e., TFLite), our method can accelerate model inference by 21.0% and 24.3% on x86-64 and ARM64 platforms, respectively. Most importantly, it can significantly reduce memory consumption by 68.8% and 36.0% on x86-64 and ARM64 platforms, respectively.



## **30. Secure Control of Connected and Automated Vehicles Using Trust-Aware Robust Event-Triggered Control Barrier Functions**

eess.SY

arXiv admin note: substantial text overlap with arXiv:2305.16818

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2401.02306v3) [paper-pdf](http://arxiv.org/pdf/2401.02306v3)

**Authors**: H M Sabbir Ahmad, Ehsan Sabouni, Akua Dickson, Wei Xiao, Christos G. Cassandras, Wenchao Li

**Abstract**: We address the security of a network of Connected and Automated Vehicles (CAVs) cooperating to safely navigate through a conflict area (e.g., traffic intersections, merging roadways, roundabouts). Previous studies have shown that such a network can be targeted by adversarial attacks causing traffic jams or safety violations ending in collisions. We focus on attacks targeting the V2X communication network used to share vehicle data and consider as well uncertainties due to noise in sensor measurements and communication channels. To combat these, motivated by recent work on the safe control of CAVs, we propose a trust-aware robust event-triggered decentralized control and coordination framework that can provably guarantee safety. We maintain a trust metric for each vehicle in the network computed based on their behavior and used to balance the tradeoff between conservativeness (when deeming every vehicle as untrustworthy) and guaranteed safety and security. It is important to highlight that our framework is invariant to the specific choice of the trust framework. Based on this framework, we propose an attack detection and mitigation scheme which has twofold benefits: (i) the trust framework is immune to false positives, and (ii) it provably guarantees safety against false positive cases. We use extensive simulations (in SUMO and CARLA) to validate the theoretical guarantees and demonstrate the efficacy of our proposed scheme to detect and mitigate adversarial attacks.



## **31. Ensemble Adversarial Defense via Integration of Multiple Dispersed Low Curvature Models**

cs.LG

Accepted to The 2024 International Joint Conference on Neural  Networks (IJCNN)

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16405v1) [paper-pdf](http://arxiv.org/pdf/2403.16405v1)

**Authors**: Kaikang Zhao, Xi Chen, Wei Huang, Liuxin Ding, Xianglong Kong, Fan Zhang

**Abstract**: The integration of an ensemble of deep learning models has been extensively explored to enhance defense against adversarial attacks. The diversity among sub-models increases the attack cost required to deceive the majority of the ensemble, thereby improving the adversarial robustness. While existing approaches mainly center on increasing diversity in feature representations or dispersion of first-order gradients with respect to input, the limited correlation between these diversity metrics and adversarial robustness constrains the performance of ensemble adversarial defense. In this work, we aim to enhance ensemble diversity by reducing attack transferability. We identify second-order gradients, which depict the loss curvature, as a key factor in adversarial robustness. Computing the Hessian matrix involved in second-order gradients is computationally expensive. To address this, we approximate the Hessian-vector product using differential approximation. Given that low curvature provides better robustness, our ensemble model was designed to consider the influence of curvature among different sub-models. We introduce a novel regularizer to train multiple more-diverse low-curvature network models. Extensive experiments across various datasets demonstrate that our ensemble model exhibits superior robustness against a range of attacks, underscoring the effectiveness of our approach.



## **32. Generating Potent Poisons and Backdoors from Scratch with Guided Diffusion**

cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16365v1) [paper-pdf](http://arxiv.org/pdf/2403.16365v1)

**Authors**: Hossein Souri, Arpit Bansal, Hamid Kazemi, Liam Fowl, Aniruddha Saha, Jonas Geiping, Andrew Gordon Wilson, Rama Chellappa, Tom Goldstein, Micah Goldblum

**Abstract**: Modern neural networks are often trained on massive datasets that are web scraped with minimal human inspection. As a result of this insecure curation pipeline, an adversary can poison or backdoor the resulting model by uploading malicious data to the internet and waiting for a victim to scrape and train on it. Existing approaches for creating poisons and backdoors start with randomly sampled clean data, called base samples, and then modify those samples to craft poisons. However, some base samples may be significantly more amenable to poisoning than others. As a result, we may be able to craft more potent poisons by carefully choosing the base samples. In this work, we use guided diffusion to synthesize base samples from scratch that lead to significantly more potent poisons and backdoors than previous state-of-the-art attacks. Our Guided Diffusion Poisoning (GDP) base samples can be combined with any downstream poisoning or backdoor attack to boost its effectiveness. Our implementation code is publicly available at: https://github.com/hsouri/GDP .



## **33. Subspace Defense: Discarding Adversarial Perturbations by Learning a Subspace for Clean Signals**

cs.LG

Accepted by COLING 2024

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.16176v1) [paper-pdf](http://arxiv.org/pdf/2403.16176v1)

**Authors**: Rui Zheng, Yuhao Zhou, Zhiheng Xi, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Deep neural networks (DNNs) are notoriously vulnerable to adversarial attacks that place carefully crafted perturbations on normal examples to fool DNNs. To better understand such attacks, a characterization of the features carried by adversarial examples is needed. In this paper, we tackle this challenge by inspecting the subspaces of sample features through spectral analysis. We first empirically show that the features of either clean signals or adversarial perturbations are redundant and span in low-dimensional linear subspaces respectively with minimal overlap, and the classical low-dimensional subspace projection can suppress perturbation features out of the subspace of clean signals. This makes it possible for DNNs to learn a subspace where only features of clean signals exist while those of perturbations are discarded, which can facilitate the distinction of adversarial examples. To prevent the residual perturbations that is inevitable in subspace learning, we propose an independence criterion to disentangle clean signals from perturbations. Experimental results show that the proposed strategy enables the model to inherently suppress adversaries, which not only boosts model robustness but also motivates new directions of effective adversarial defense.



## **34. Large Language Models for Blockchain Security: A Systematic Literature Review**

cs.CR

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.14280v2) [paper-pdf](http://arxiv.org/pdf/2403.14280v2)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.



## **35. ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations**

cs.LG

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2308.10457v5) [paper-pdf](http://arxiv.org/pdf/2308.10457v5)

**Authors**: Xinpeng Ling, Jie Fu, Kuncan Wang, Haitao Liu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks.   We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication rounds are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DPSGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of Differentially Private Federated Learning with Adaptive Local Iterations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and Cifar10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. Code is available at https://github.com/KnightWan/ALI-DPFL.



## **36. Robust Diffusion Models for Adversarial Purification**

cs.CV

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.16067v1) [paper-pdf](http://arxiv.org/pdf/2403.16067v1)

**Authors**: Guang Lin, Zerui Tao, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also mitigate the accuracy-robustness trade-off of DMs for the first time, which also provides DM-based AP an efficient adaptive ability to new attacks. Extensive experiments are conducted to demonstrate that our method achieves the state-of-the-art results and exhibits generalization against different attacks.



## **37. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

cs.CV

Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2310.11868v2) [paper-pdf](http://arxiv.org/pdf/2310.11868v2)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models.Through extensive benchmarking, we evaluate the robustness of five widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safety-driven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: This paper contains model outputs that may be offensive in nature.



## **38. An Embarrassingly Simple Defense Against Backdoor Attacks On SSL**

cs.CV

10 pages, 5 figures

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.15918v1) [paper-pdf](http://arxiv.org/pdf/2403.15918v1)

**Authors**: Aryan Satpathy, Nilaksh, Dhruva Rajwade

**Abstract**: Self Supervised Learning (SSL) has emerged as a powerful paradigm to tackle data landscapes with absence of human supervision. The ability to learn meaningful tasks without the use of labeled data makes SSL a popular method to manage large chunks of data in the absence of labels. However, recent work indicates SSL to be vulnerable to backdoor attacks, wherein models can be controlled, possibly maliciously, to suit an adversary's motives. Li et.al (2022) introduce a novel frequency-based backdoor attack: CTRL. They show that CTRL can be used to efficiently and stealthily gain control over a victim's model trained using SSL. In this work, we devise two defense strategies against frequency-based attacks in SSL: One applicable before model training and the second to be applied during model inference. Our first contribution utilizes the invariance property of the downstream task to defend against backdoor attacks in a generalizable fashion. We observe the ASR (Attack Success Rate) to reduce by over 60% across experiments. Our Inference-time defense relies on evasiveness of the attack and uses the luminance channel to defend against attacks. Using object classification as the downstream task for SSL, we demonstrate successful defense strategies that do not require re-training of the model. Code is available at https://github.com/Aryan-Satpathy/Backdoor.



## **39. Effect of Ambient-Intrinsic Dimension Gap on Adversarial Vulnerability**

cs.LG

AISTATS 2024

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.03967v2) [paper-pdf](http://arxiv.org/pdf/2403.03967v2)

**Authors**: Rajdeep Haldar, Yue Xing, Qifan Song

**Abstract**: The existence of adversarial attacks on machine learning models imperceptible to a human is still quite a mystery from a theoretical perspective. In this work, we introduce two notions of adversarial attacks: natural or on-manifold attacks, which are perceptible by a human/oracle, and unnatural or off-manifold attacks, which are not. We argue that the existence of the off-manifold attacks is a natural consequence of the dimension gap between the intrinsic and ambient dimensions of the data. For 2-layer ReLU networks, we prove that even though the dimension gap does not affect generalization performance on samples drawn from the observed data space, it makes the clean-trained model more vulnerable to adversarial perturbations in the off-manifold direction of the data space. Our main results provide an explicit relationship between the $\ell_2,\ell_{\infty}$ attack strength of the on/off-manifold attack and the dimension gap.



## **40. Adversarial Defense Teacher for Cross-Domain Object Detection under Poor Visibility Conditions**

cs.CV

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.15786v1) [paper-pdf](http://arxiv.org/pdf/2403.15786v1)

**Authors**: Kaiwen Wang, Yinzhe Shen, Martin Lauer

**Abstract**: Existing object detectors encounter challenges in handling domain shifts between training and real-world data, particularly under poor visibility conditions like fog and night. Cutting-edge cross-domain object detection methods use teacher-student frameworks and compel teacher and student models to produce consistent predictions under weak and strong augmentations, respectively. In this paper, we reveal that manually crafted augmentations are insufficient for optimal teaching and present a simple yet effective framework named Adversarial Defense Teacher (ADT), leveraging adversarial defense to enhance teaching quality. Specifically, we employ adversarial attacks, encouraging the model to generalize on subtly perturbed inputs that effectively deceive the model. To address small objects under poor visibility conditions, we propose a Zoom-in Zoom-out strategy, which zooms-in images for better pseudo-labels and zooms-out images and pseudo-labels to learn refined features. Our results demonstrate that ADT achieves superior performance, reaching 54.5% mAP on Foggy Cityscapes, surpassing the previous state-of-the-art by 2.6% mAP.



## **41. Breaking Down the Defenses: A Comparative Survey of Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.04786v2) [paper-pdf](http://arxiv.org/pdf/2403.04786v2)

**Authors**: Arijit Ghosh Chowdhury, Md Mofijul Islam, Vaibhav Kumar, Faysal Hossain Shezan, Vaibhav Kumar, Vinija Jain, Aman Chadha

**Abstract**: Large Language Models (LLMs) have become a cornerstone in the field of Natural Language Processing (NLP), offering transformative capabilities in understanding and generating human-like text. However, with their rising prominence, the security and vulnerability aspects of these models have garnered significant attention. This paper presents a comprehensive survey of the various forms of attacks targeting LLMs, discussing the nature and mechanisms of these attacks, their potential impacts, and current defense strategies. We delve into topics such as adversarial attacks that aim to manipulate model outputs, data poisoning that affects model training, and privacy concerns related to training data exploitation. The paper also explores the effectiveness of different attack methodologies, the resilience of LLMs against these attacks, and the implications for model integrity and user trust. By examining the latest research, we provide insights into the current landscape of LLM vulnerabilities and defense mechanisms. Our objective is to offer a nuanced understanding of LLM attacks, foster awareness within the AI community, and inspire robust solutions to mitigate these risks in future developments.



## **42. On the Privacy Effect of Data Enhancement via the Lens of Memorization**

cs.LG

Accepted by IEEE TIFS, 17 pages

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2208.08270v4) [paper-pdf](http://arxiv.org/pdf/2208.08270v4)

**Authors**: Xiao Li, Qiongxiu Li, Zhanhao Hu, Xiaolin Hu

**Abstract**: Machine learning poses severe privacy concerns as it has been shown that the learned models can reveal sensitive information about their training data. Many works have investigated the effect of widely adopted data augmentation and adversarial training techniques, termed data enhancement in the paper, on the privacy leakage of machine learning models. Such privacy effects are often measured by membership inference attacks (MIAs), which aim to identify whether a particular example belongs to the training set or not. We propose to investigate privacy from a new perspective called memorization. Through the lens of memorization, we find that previously deployed MIAs produce misleading results as they are less likely to identify samples with higher privacy risks as members compared to samples with low privacy risks. To solve this problem, we deploy a recent attack that can capture individual samples' memorization degrees for evaluation. Through extensive experiments, we unveil several findings about the connections between three essential properties of machine learning models, including privacy, generalization gap, and adversarial robustness. We demonstrate that the generalization gap and privacy leakage are less correlated than those of the previous results. Moreover, there is not necessarily a trade-off between adversarial robustness and privacy as stronger adversarial robustness does not make the model more susceptible to privacy attacks.



## **43. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

cs.CR

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.05030v2) [paper-pdf](http://arxiv.org/pdf/2403.05030v2)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: AI systems sometimes exhibit harmful unintended behaviors post-deployment. This is often despite extensive diagnostics and debugging by developers. Minimizing risks from models is challenging because the attack surface is so large. It is not tractable to exhaustively search for inputs that may cause a model to fail. Red-teaming and adversarial training (AT) are commonly used to make AI systems more robust. However, they have not been sufficient to avoid many real-world failure modes that differ from the ones adversarially trained on. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.



## **44. From Hardware Fingerprint to Access Token: Enhancing the Authentication on IoT Devices**

cs.CR

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15271v1) [paper-pdf](http://arxiv.org/pdf/2403.15271v1)

**Authors**: Yue Xiao, Yi He, Xiaoli Zhang, Qian Wang, Renjie Xie, Kun Sun, Ke Xu, Qi Li

**Abstract**: The proliferation of consumer IoT products in our daily lives has raised the need for secure device authentication and access control. Unfortunately, these resource-constrained devices typically use token-based authentication, which is vulnerable to token compromise attacks that allow attackers to impersonate the devices and perform malicious operations by stealing the access token. Using hardware fingerprints to secure their authentication is a promising way to mitigate these threats. However, once attackers have stolen some hardware fingerprints (e.g., via MitM attacks), they can bypass the hardware authentication by training a machine learning model to mimic fingerprints or reusing these fingerprints to craft forge requests.   In this paper, we present MCU-Token, a secure hardware fingerprinting framework for MCU-based IoT devices even if the cryptographic mechanisms (e.g., private keys) are compromised. MCU-Token can be easily integrated with various IoT devices by simply adding a short hardware fingerprint-based token to the existing payload. To prevent the reuse of this token, we propose a message mapping approach that binds the token to a specific request via generating the hardware fingerprints based on the request payload. To defeat the machine learning attacks, we mix the valid fingerprints with poisoning data so that attackers cannot train a usable model with the leaked tokens. MCU-Token can defend against armored adversary who may replay, craft, and offload the requests via MitM or use both hardware (e.g., use identical devices) and software (e.g., machine learning attacks) strategies to mimic the fingerprints. The system evaluation shows that MCU-Token can achieve high accuracy (over 97%) with a low overhead across various IoT devices and application scenarios.



## **45. Robust optimization for adversarial learning with finite sample complexity guarantees**

cs.LG

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15207v1) [paper-pdf](http://arxiv.org/pdf/2403.15207v1)

**Authors**: André Bertolace, Konstatinos Gatsis, Kostas Margellos

**Abstract**: Decision making and learning in the presence of uncertainty has attracted significant attention in view of the increasing need to achieve robust and reliable operations. In the case where uncertainty stems from the presence of adversarial attacks this need is becoming more prominent. In this paper we focus on linear and nonlinear classification problems and propose a novel adversarial training method for robust classifiers, inspired by Support Vector Machine (SVM) margins. We view robustness under a data driven lens, and derive finite sample complexity bounds for both linear and non-linear classifiers in binary and multi-class scenarios. Notably, our bounds match natural classifiers' complexity. Our algorithm minimizes a worst-case surrogate loss using Linear Programming (LP) and Second Order Cone Programming (SOCP) for linear and non-linear models. Numerical experiments on the benchmark MNIST and CIFAR10 datasets show our approach's comparable performance to state-of-the-art methods, without needing adversarial examples during training. Our work offers a comprehensive framework for enhancing binary linear and non-linear classifier robustness, embedding robustness in learning under the presence of adversaries.



## **46. Privacy-Preserving End-to-End Spoken Language Understanding**

cs.CR

Accepted by IJCAI

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15510v1) [paper-pdf](http://arxiv.org/pdf/2403.15510v1)

**Authors**: Yinggui Wang, Wei Huang, Le Yang

**Abstract**: Spoken language understanding (SLU), one of the key enabling technologies for human-computer interaction in IoT devices, provides an easy-to-use user interface. Human speech can contain a lot of user-sensitive information, such as gender, identity, and sensitive content. New types of security and privacy breaches have thus emerged. Users do not want to expose their personal sensitive information to malicious attacks by untrusted third parties. Thus, the SLU system needs to ensure that a potential malicious attacker cannot deduce the sensitive attributes of the users, while it should avoid greatly compromising the SLU accuracy. To address the above challenge, this paper proposes a novel SLU multi-task privacy-preserving model to prevent both the speech recognition (ASR) and identity recognition (IR) attacks. The model uses the hidden layer separation technique so that SLU information is distributed only in a specific portion of the hidden layer, and the other two types of information are removed to obtain a privacy-secure hidden layer. In order to achieve good balance between efficiency and privacy, we introduce a new mechanism of model pre-training, namely joint adversarial training, to further enhance the user privacy. Experiments over two SLU datasets show that the proposed method can reduce the accuracy of both the ASR and IR attacks close to that of a random guess, while leaving the SLU performance largely unaffected.



## **47. TTPXHunter: Actionable Threat Intelligence Extraction as TTPs from Finished Cyber Threat Reports**

cs.CR

Under Review

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.03267v3) [paper-pdf](http://arxiv.org/pdf/2403.03267v3)

**Authors**: Nanda Rani, Bikash Saha, Vikas Maurya, Sandeep Kumar Shukla

**Abstract**: Understanding the modus operandi of adversaries aids organizations in employing efficient defensive strategies and sharing intelligence in the community. This knowledge is often present in unstructured natural language text within threat analysis reports. A translation tool is needed to interpret the modus operandi explained in the sentences of the threat report and translate it into a structured format. This research introduces a methodology named TTPXHunter for the automated extraction of threat intelligence in terms of Tactics, Techniques, and Procedures (TTPs) from finished cyber threat reports. It leverages cyber domain-specific state-of-the-art natural language processing (NLP) to augment sentences for minority class TTPs and refine pinpointing the TTPs in threat analysis reports significantly. The knowledge of threat intelligence in terms of TTPs is essential for comprehensively understanding cyber threats and enhancing detection and mitigation strategies. We create two datasets: an augmented sentence-TTP dataset of 39,296 samples and a 149 real-world cyber threat intelligence report-to-TTP dataset. Further, we evaluate TTPXHunter on the augmented sentence dataset and the cyber threat reports. The TTPXHunter achieves the highest performance of 92.42% f1-score on the augmented dataset, and it also outperforms existing state-of-the-art solutions in TTP extraction by achieving an f1-score of 97.09% when evaluated over the report dataset. TTPXHunter significantly improves cybersecurity threat intelligence by offering quick, actionable insights into attacker behaviors. This advancement automates threat intelligence analysis, providing a crucial tool for cybersecurity professionals fighting cyber threats.



## **48. Diffusion Attack: Leveraging Stable Diffusion for Naturalistic Image Attacking**

cs.CV

Accepted to IEEE VRW

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14778v1) [paper-pdf](http://arxiv.org/pdf/2403.14778v1)

**Authors**: Qianyu Guo, Jiaming Fu, Yawen Lu, Dongming Gan

**Abstract**: In Virtual Reality (VR), adversarial attack remains a significant security threat. Most deep learning-based methods for physical and digital adversarial attacks focus on enhancing attack performance by crafting adversarial examples that contain large printable distortions that are easy for human observers to identify. However, attackers rarely impose limitations on the naturalness and comfort of the appearance of the generated attack image, resulting in a noticeable and unnatural attack. To address this challenge, we propose a framework to incorporate style transfer to craft adversarial inputs of natural styles that exhibit minimal detectability and maximum natural appearance, while maintaining superior attack capabilities.



## **49. Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures**

cs.CV

32 pages, 15 Tables, and 9 Figures

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14772v1) [paper-pdf](http://arxiv.org/pdf/2403.14772v1)

**Authors**: Sayanton V. Dibbo, Adam Breuer, Juston Moore, Michael Teti

**Abstract**: Recent model inversion attack algorithms permit adversaries to reconstruct a neural network's private training data just by repeatedly querying the network and inspecting its outputs. In this work, we develop a novel network architecture that leverages sparse-coding layers to obtain superior robustness to this class of attacks. Three decades of computer science research has studied sparse coding in the context of image denoising, object recognition, and adversarial misclassification settings, but to the best of our knowledge, its connection to state-of-the-art privacy vulnerabilities remains unstudied. However, sparse coding architectures suggest an advantageous means to defend against model inversion attacks because they allow us to control the amount of irrelevant private information encoded in a network's intermediate representations in a manner that can be computed efficiently during training and that is known to have little effect on classification accuracy. Specifically, compared to networks trained with a variety of state-of-the-art defenses, our sparse-coding architectures maintain comparable or higher classification accuracy while degrading state-of-the-art training data reconstructions by factors of 1.1 to 18.3 across a variety of reconstruction quality metrics (PSNR, SSIM, FID). This performance advantage holds across 5 datasets ranging from CelebA faces to medical images and CIFAR-10, and across various state-of-the-art SGD-based and GAN-based inversion attacks, including Plug-&-Play attacks. We provide a cluster-ready PyTorch codebase to promote research and standardize defense evaluations.



## **50. TMI! Finetuned Models Leak Private Information from their Pretraining Data**

cs.LG

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2306.01181v2) [paper-pdf](http://arxiv.org/pdf/2306.01181v2)

**Authors**: John Abascal, Stanley Wu, Alina Oprea, Jonathan Ullman

**Abstract**: Transfer learning has become an increasingly popular technique in machine learning as a way to leverage a pretrained model trained for one task to assist with building a finetuned model for a related task. This paradigm has been especially popular for $\textit{privacy}$ in machine learning, where the pretrained model is considered public, and only the data for finetuning is considered sensitive. However, there are reasons to believe that the data used for pretraining is still sensitive, making it essential to understand how much information the finetuned model leaks about the pretraining data. In this work we propose a new membership-inference threat model where the adversary only has access to the finetuned model and would like to infer the membership of the pretraining data. To realize this threat model, we implement a novel metaclassifier-based attack, $\textbf{TMI}$, that leverages the influence of memorized pretraining samples on predictions in the downstream task. We evaluate $\textbf{TMI}$ on both vision and natural language tasks across multiple transfer learning settings, including finetuning with differential privacy. Through our evaluation, we find that $\textbf{TMI}$ can successfully infer membership of pretraining examples using query access to the finetuned model. An open-source implementation of $\textbf{TMI}$ can be found $\href{https://github.com/johnmath/tmi-pets24}{\text{on GitHub}}$.



