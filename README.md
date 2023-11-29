# Latest Adversarial Attack Papers
**update at 2023-11-29 10:29:29**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Scalable Extraction of Training Data from (Production) Language Models**

cs.LG

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.17035v1) [paper-pdf](http://arxiv.org/pdf/2311.17035v1)

**Authors**: Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, Katherine Lee

**Abstract**: This paper studies extractable memorization: training data that an adversary can efficiently extract by querying a machine learning model without prior knowledge of the training dataset. We show an adversary can extract gigabytes of training data from open-source language models like Pythia or GPT-Neo, semi-open models like LLaMA or Falcon, and closed models like ChatGPT. Existing techniques from the literature suffice to attack unaligned models; in order to attack the aligned ChatGPT, we develop a new divergence attack that causes the model to diverge from its chatbot-style generations and emit training data at a rate 150x higher than when behaving properly. Our methods show practical attacks can recover far more data than previously thought, and reveal that current alignment techniques do not eliminate memorization.



## **2. Breaking Boundaries: Balancing Performance and Robustness in Deep Wireless Traffic Forecasting**

cs.LG

Accepted for presentation at the ARTMAN workshop, part of the ACM  Conference on Computer and Communications Security (CCS), 2023

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.09790v3) [paper-pdf](http://arxiv.org/pdf/2311.09790v3)

**Authors**: Romain Ilbert, Thai V. Hoang, Zonghua Zhang, Themis Palpanas

**Abstract**: Balancing the trade-off between accuracy and robustness is a long-standing challenge in time series forecasting. While most of existing robust algorithms have achieved certain suboptimal performance on clean data, sustaining the same performance level in the presence of data perturbations remains extremely hard. In this paper, we study a wide array of perturbation scenarios and propose novel defense mechanisms against adversarial attacks using real-world telecom data. We compare our strategy against two existing adversarial training algorithms under a range of maximal allowed perturbations, defined using $\ell_{\infty}$-norm, $\in [0.1,0.4]$. Our findings reveal that our hybrid strategy, which is composed of a classifier to detect adversarial examples, a denoiser to eliminate noise from the perturbed data samples, and a standard forecaster, achieves the best performance on both clean and perturbed data. Our optimal model can retain up to $92.02\%$ the performance of the original forecasting model in terms of Mean Squared Error (MSE) on clean data, while being more robust than the standard adversarially trained models on perturbed data. Its MSE is 2.71$\times$ and 2.51$\times$ lower than those of comparing methods on normal and perturbed data, respectively. In addition, the components of our models can be trained in parallel, resulting in better computational efficiency. Our results indicate that we can optimally balance the trade-off between the performance and robustness of forecasting models by improving the classifier and denoiser, even in the presence of sophisticated and destructive poisoning attacks.



## **3. Generation of Games for Opponent Model Differentiation**

cs.AI

4 pages

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16781v1) [paper-pdf](http://arxiv.org/pdf/2311.16781v1)

**Authors**: David Milec, Viliam Lisý, Christopher Kiekintveld

**Abstract**: Protecting against adversarial attacks is a common multiagent problem. Attackers in the real world are predominantly human actors, and the protection methods often incorporate opponent models to improve the performance when facing humans. Previous results show that modeling human behavior can significantly improve the performance of the algorithms. However, modeling humans correctly is a complex problem, and the models are often simplified and assume humans make mistakes according to some distribution or train parameters for the whole population from which they sample. In this work, we use data gathered by psychologists who identified personality types that increase the likelihood of performing malicious acts. However, in the previous work, the tests on a handmade game could not show strategic differences between the models. We created a novel model that links its parameters to psychological traits. We optimized over parametrized games and created games in which the differences are profound. Our work can help with automatic game generation when we need a game in which some models will behave differently and to identify situations in which the models do not align.



## **4. Cooperative Abnormal Node Detection with Adversary Resistance: A Probabilistic Approach**

eess.SY

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16661v1) [paper-pdf](http://arxiv.org/pdf/2311.16661v1)

**Authors**: Yingying Huangfu, Tian Bai

**Abstract**: This paper presents a novel probabilistic detection scheme called Cooperative Statistical Detection (CSD) for abnormal node detection while defending against adversarial attacks in cluster-tree networks. The CSD performs a two-phase process: 1) designing a likelihood ratio test (LRT) for a non-root node at its children from the perspective of packet loss; 2) making an overall decision at the root node based on the aggregated detection data of the nodes over tree branches. In most adversarial scenarios, malicious children knowing the detection policy can generate falsified data to protect the abnormal parent from being detected or frame its normal parent as an anomalous node. To resolve this issue, a modified Z-score-based falsification-resistant mechanism is presented in the CSD to remove untrustworthy information. Through theoretical analysis, we show that the LRT-based method achieves perfect detection, i.e., both the false alarm and missed detection probabilities decay exponentially to zero. Furthermore, the optimal removal threshold of the modified Z-score method is derived for falsifications with uncertain strategies and guarantees perfect detection of the CSD. As our simulation results show, the CSD approach is robust to falsifications and can rapidly reach $99\%$ detection accuracy, even in existing adversarial scenarios, which outperforms state-of-the-art technology.



## **5. On the Role of Randomization in Adversarially Robust Classification**

cs.LG

10 pages main paper (27 total), 2 figures in main paper. Neurips 2023

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2302.07221v3) [paper-pdf](http://arxiv.org/pdf/2302.07221v3)

**Authors**: Lucas Gnecco-Heredia, Yann Chevaleyre, Benjamin Negrevergne, Laurent Meunier, Muni Sreenivas Pydi

**Abstract**: Deep neural networks are known to be vulnerable to small adversarial perturbations in test data. To defend against adversarial attacks, probabilistic classifiers have been proposed as an alternative to deterministic ones. However, literature has conflicting findings on the effectiveness of probabilistic classifiers in comparison to deterministic ones. In this paper, we clarify the role of randomization in building adversarially robust classifiers. Given a base hypothesis set of deterministic classifiers, we show the conditions under which a randomized ensemble outperforms the hypothesis set in adversarial risk, extending previous results. Additionally, we show that for any probabilistic binary classifier (including randomized ensembles), there exists a deterministic classifier that outperforms it. Finally, we give an explicit description of the deterministic hypothesis set that contains such a deterministic classifier for many types of commonly used probabilistic classifiers, i.e. randomized ensembles and parametric/input noise injection.



## **6. Efficient Key-Based Adversarial Defense for ImageNet by Using Pre-trained Model**

cs.CV

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16577v1) [paper-pdf](http://arxiv.org/pdf/2311.16577v1)

**Authors**: AprilPyone MaungMaung, Isao Echizen, Hitoshi Kiya

**Abstract**: In this paper, we propose key-based defense model proliferation by leveraging pre-trained models and utilizing recent efficient fine-tuning techniques on ImageNet-1k classification. First, we stress that deploying key-based models on edge devices is feasible with the latest model deployment advancements, such as Apple CoreML, although the mainstream enterprise edge artificial intelligence (Edge AI) has been focused on the Cloud. Then, we point out that the previous key-based defense on on-device image classification is impractical for two reasons: (1) training many classifiers from scratch is not feasible, and (2) key-based defenses still need to be thoroughly tested on large datasets like ImageNet. To this end, we propose to leverage pre-trained models and utilize efficient fine-tuning techniques to proliferate key-based models even on limited computing resources. Experiments were carried out on the ImageNet-1k dataset using adaptive and non-adaptive attacks. The results show that our proposed fine-tuned key-based models achieve a superior classification accuracy (more than 10% increase) compared to the previous key-based models on classifying clean and adversarial examples.



## **7. On the Robustness of Decision-Focused Learning**

cs.LG

17 pages, 45 figures, submitted to AAAI artificial intelligence for  operations research workshop

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16487v1) [paper-pdf](http://arxiv.org/pdf/2311.16487v1)

**Authors**: Yehya Farhat

**Abstract**: Decision-Focused Learning (DFL) is an emerging learning paradigm that tackles the task of training a machine learning (ML) model to predict missing parameters of an incomplete optimization problem, where the missing parameters are predicted. DFL trains an ML model in an end-to-end system, by integrating the prediction and optimization tasks, providing better alignment of the training and testing objectives. DFL has shown a lot of promise and holds the capacity to revolutionize decision-making in many real-world applications. However, very little is known about the performance of these models under adversarial attacks. We adopt ten unique DFL methods and benchmark their performance under two distinctly focused attacks adapted towards the Predict-then-Optimize problem setting. Our study proposes the hypothesis that the robustness of a model is highly correlated with its ability to find predictions that lead to optimal decisions without deviating from the ground-truth label. Furthermore, we provide insight into how to target the models that violate this condition and show how these models respond differently depending on the achieved optimality at the end of their training cycles.



## **8. Threshold Breaker: Can Counter-Based RowHammer Prevention Mechanisms Truly Safeguard DRAM?**

cs.AR

7 pages, 6 figures

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2311.16460v1) [paper-pdf](http://arxiv.org/pdf/2311.16460v1)

**Authors**: Ranyang Zhou, Jacqueline Liu, Sabbir Ahmed, Nakul Kochar, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: This paper challenges the existing victim-focused counter-based RowHammer detection mechanisms by experimentally demonstrating a novel multi-sided fault injection attack technique called Threshold Breaker. This mechanism can effectively bypass the most advanced counter-based defense mechanisms by soft-attacking the rows at a farther physical distance from the target rows. While no prior work has demonstrated the effect of such an attack, our work closes this gap by systematically testing 128 real commercial DDR4 DRAM products and reveals that the Threshold Breaker affects various chips from major DRAM manufacturers. As a case study, we compare the performance efficiency between our mechanism and a well-known double-sided attack by performing adversarial weight attacks on a modern Deep Neural Network (DNN). The results demonstrate that the Threshold Breaker can deliberately deplete the intelligence of the targeted DNN system while DRAM is fully protected.



## **9. Certifying LLM Safety against Adversarial Prompting**

cs.CL

**SubmitDate**: 2023-11-28    [abs](http://arxiv.org/abs/2309.02705v2) [paper-pdf](http://arxiv.org/pdf/2309.02705v2)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial attacks, which add maliciously designed token sequences to a harmful prompt to bypass the model's safety guards. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 92% of harmful prompts and labels 94% of safe prompts correctly using the open-source language model Llama 2 as the safety filter. We further improve the filter's performance, in terms of accuracy and speed, by replacing Llama 2 with a DistilBERT safety classifier fine-tuned on safe and harmful prompts. Additionally, we propose two efficient empirical defenses: i) RandEC, a randomized version of erase-and-check that evaluates the safety filter on a small subset of the erased subsequences, and ii) GradEC, a gradient-based version that optimizes the erased tokens to remove the adversarial sequence. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.



## **10. Mate! Are You Really Aware? An Explainability-Guided Testing Framework for Robustness of Malware Detectors**

cs.CR

Accepted at ESEC/FSE 2023. https://doi.org/10.1145/3611643.3616309

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2111.10085v4) [paper-pdf](http://arxiv.org/pdf/2111.10085v4)

**Authors**: Ruoxi Sun, Minhui Xue, Gareth Tyson, Tian Dong, Shaofeng Li, Shuo Wang, Haojin Zhu, Seyit Camtepe, Surya Nepal

**Abstract**: Numerous open-source and commercial malware detectors are available. However, their efficacy is threatened by new adversarial attacks, whereby malware attempts to evade detection, e.g., by performing feature-space manipulation. In this work, we propose an explainability-guided and model-agnostic testing framework for robustness of malware detectors when confronted with adversarial attacks. The framework introduces the concept of Accrued Malicious Magnitude (AMM) to identify which malware features could be manipulated to maximize the likelihood of evading detection. We then use this framework to test several state-of-the-art malware detectors' abilities to detect manipulated malware. We find that (i) commercial antivirus engines are vulnerable to AMM-guided test cases; (ii) the ability of a manipulated malware generated using one detector to evade detection by another detector (i.e., transferability) depends on the overlap of features with large AMM values between the different detectors; and (iii) AMM values effectively measure the fragility of features (i.e., capability of feature-space manipulation to flip the prediction results) and explain the robustness of malware detectors facing evasion attacks. Our findings shed light on the limitations of current malware detectors, as well as how they can be improved.



## **11. How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs**

cs.CV

H.T., C.C., and Z.W. contribute equally. Work done during H.T. and  Z.W.'s internship at UCSC, and C.C. and Y.Z.'s internship at UNC

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.16101v1) [paper-pdf](http://arxiv.org/pdf/2311.16101v1)

**Authors**: Haoqin Tu, Chenhang Cui, Zijun Wang, Yiyang Zhou, Bingchen Zhao, Junlin Han, Wangchunshu Zhou, Huaxiu Yao, Cihang Xie

**Abstract**: This work focuses on the potential of Vision LLMs (VLLMs) in visual reasoning. Different from prior studies, we shift our focus from evaluating standard performance to introducing a comprehensive safety evaluation suite, covering both out-of-distribution (OOD) generalization and adversarial robustness. For the OOD evaluation, we present two novel VQA datasets, each with one variant, designed to test model performance under challenging conditions. In exploring adversarial robustness, we propose a straightforward attack strategy for misleading VLLMs to produce visual-unrelated responses. Moreover, we assess the efficacy of two jailbreaking strategies, targeting either the vision or language component of VLLMs. Our evaluation of 21 diverse models, ranging from open-source VLLMs to GPT-4V, yields interesting observations: 1) Current VLLMs struggle with OOD texts but not images, unless the visual information is limited; and 2) These VLLMs can be easily misled by deceiving vision encoders only, and their vision-language training often compromise safety protocols. We release this safety evaluation suite at https://github.com/UCSC-VLAA/vllm-safety-benchmark.



## **12. Adversaral Doodles: Interpretable and Human-drawable Attacks Provide Describable Insights**

cs.CV

Submitted to CVPR 2024

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15994v1) [paper-pdf](http://arxiv.org/pdf/2311.15994v1)

**Authors**: Ryoya Nara, Yusuke Matsui

**Abstract**: DNN-based image classification models are susceptible to adversarial attacks. Most previous adversarial attacks do not focus on the interpretability of the generated adversarial examples, and we cannot gain insights into the mechanism of the target classifier from the attacks. Therefore, we propose Adversarial Doodles, which have interpretable shapes. We optimize black b\'ezier curves to fool the target classifier by overlaying them onto the input image. By introducing random perspective transformation and regularizing the doodled area, we obtain compact attacks that cause misclassification even when humans replicate them by hand. Adversarial doodles provide describable and intriguing insights into the relationship between our attacks and the classifier's output. We utilize adversarial doodles and discover the bias inherent in the target classifier, such as "We add two strokes on its head, a triangle onto its body, and two lines inside the triangle on a bird image. Then, the classifier misclassifies the image as a butterfly."



## **13. CALICO: Self-Supervised Camera-LiDAR Contrastive Pre-training for BEV Perception**

cs.CV

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2306.00349v2) [paper-pdf](http://arxiv.org/pdf/2306.00349v2)

**Authors**: Jiachen Sun, Haizhong Zheng, Qingzhao Zhang, Atul Prakash, Z. Morley Mao, Chaowei Xiao

**Abstract**: Perception is crucial in the realm of autonomous driving systems, where bird's eye view (BEV)-based architectures have recently reached state-of-the-art performance. The desirability of self-supervised representation learning stems from the expensive and laborious process of annotating 2D and 3D data. Although previous research has investigated pretraining methods for both LiDAR and camera-based 3D object detection, a unified pretraining framework for multimodal BEV perception is missing. In this study, we introduce CALICO, a novel framework that applies contrastive objectives to both LiDAR and camera backbones. Specifically, CALICO incorporates two stages: point-region contrast (PRC) and region-aware distillation (RAD). PRC better balances the region- and scene-level representation learning on the LiDAR modality and offers significant performance improvement compared to existing methods. RAD effectively achieves contrastive distillation on our self-trained teacher model. CALICO's efficacy is substantiated by extensive evaluations on 3D object detection and BEV map segmentation tasks, where it delivers significant performance improvements. Notably, CALICO outperforms the baseline method by 10.5% and 8.6% on NDS and mAP. Moreover, CALICO boosts the robustness of multimodal 3D object detection against adversarial attacks and corruption. Additionally, our framework can be tailored to different backbones and heads, positioning it as a promising approach for multimodal BEV perception.



## **14. AdaptGuard: Defending Against Universal Attacks for Model Adaptation**

cs.CR

ICCV2023

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2303.10594v2) [paper-pdf](http://arxiv.org/pdf/2303.10594v2)

**Authors**: Lijun Sheng, Jian Liang, Ran He, Zilei Wang, Tieniu Tan

**Abstract**: Model adaptation aims at solving the domain transfer problem under the constraint of only accessing the pretrained source models. With the increasing considerations of data privacy and transmission efficiency, this paradigm has been gaining recent popularity. This paper studies the vulnerability to universal attacks transferred from the source domain during model adaptation algorithms due to the existence of malicious providers. We explore both universal adversarial perturbations and backdoor attacks as loopholes on the source side and discover that they still survive in the target models after adaptation. To address this issue, we propose a model preprocessing framework, named AdaptGuard, to improve the security of model adaptation algorithms. AdaptGuard avoids direct use of the risky source parameters through knowledge distillation and utilizes the pseudo adversarial samples under adjusted radius to enhance the robustness. AdaptGuard is a plug-and-play module that requires neither robust pretrained models nor any changes for the following model adaptation algorithms. Extensive results on three commonly used datasets and two popular adaptation methods validate that AdaptGuard can effectively defend against universal attacks and maintain clean accuracy in the target domain simultaneously. We hope this research will shed light on the safety and robustness of transfer learning. Code is available at https://github.com/TomSheng21/AdaptGuard.



## **15. Attend Who is Weak: Enhancing Graph Condensation via Cross-Free Adversarial Training**

cs.LG

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15772v1) [paper-pdf](http://arxiv.org/pdf/2311.15772v1)

**Authors**: Xinglin Li, Kun Wang, Hanhui Deng, Yuxuan Liang, Di Wu

**Abstract**: In this paper, we study the \textit{graph condensation} problem by compressing the large, complex graph into a concise, synthetic representation that preserves the most essential and discriminative information of structure and features. We seminally propose the concept of Shock Absorber (a type of perturbation) that enhances the robustness and stability of the original graphs against changes in an adversarial training fashion. Concretely, (I) we forcibly match the gradients between pre-selected graph neural networks (GNNs) trained on a synthetic, simplified graph and the original training graph at regularly spaced intervals. (II) Before each update synthetic graph point, a Shock Absorber serves as a gradient attacker to maximize the distance between the synthetic dataset and the original graph by selectively perturbing the parts that are underrepresented or insufficiently informative. We iteratively repeat the above two processes (I and II) in an adversarial training fashion to maintain the highly-informative context without losing correlation with the original dataset. More importantly, our shock absorber and the synthesized graph parallelly share the backward process in a free training manner. Compared to the original adversarial training, it introduces almost no additional time overhead.   We validate our framework across 8 datasets (3 graph and 5 node classification datasets) and achieve prominent results: for example, on Cora, Citeseer and Ogbn-Arxiv, we can gain nearly 1.13% to 5.03% improvements compare with SOTA models. Moreover, our algorithm adds only about 0.2% to 2.2% additional time overhead over Flicker, Citeseer and Ogbn-Arxiv. Compared to the general adversarial training, our approach improves time efficiency by nearly 4-fold.



## **16. The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing**

cs.LG

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2309.16883v2) [paper-pdf](http://arxiv.org/pdf/2309.16883v2)

**Authors**: Blaise Delattre, Alexandre Araujo, Quentin Barthélemy, Alexandre Allauzen

**Abstract**: Real-life applications of deep neural networks are hindered by their unsteady predictions when faced with noisy inputs and adversarial attacks. The certified radius is in this context a crucial indicator of the robustness of models. However how to design an efficient classifier with a sufficient certified radius? Randomized smoothing provides a promising framework by relying on noise injection in inputs to obtain a smoothed and more robust classifier. In this paper, we first show that the variance introduced by randomized smoothing closely interacts with two other important properties of the classifier, \textit{i.e.} its Lipschitz constant and margin. More precisely, our work emphasizes the dual impact of the Lipschitz constant of the base classifier, on both the smoothed classifier and the empirical variance. Moreover, to increase the certified robust radius, we introduce a different simplex projection technique for the base classifier to leverage the variance-margin trade-off thanks to Bernstein's concentration inequality, along with an enhanced Lipschitz bound. Experimental results show a significant improvement in certified accuracy compared to current state-of-the-art methods. Our novel certification procedure allows us to use pre-trained models that are used with randomized smoothing, effectively improving the current certification radius in a zero-shot manner.



## **17. SLMIA-SR: Speaker-Level Membership Inference Attacks against Speaker Recognition Systems**

cs.CR

In Proceedings of the 31st Network and Distributed System Security  (NDSS) Symposium, 2024

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2309.07983v2) [paper-pdf](http://arxiv.org/pdf/2309.07983v2)

**Authors**: Guangke Chen, Yedi Zhang, Fu Song

**Abstract**: Membership inference attacks allow adversaries to determine whether a particular example was contained in the model's training dataset. While previous works have confirmed the feasibility of such attacks in various applications, none has focused on speaker recognition (SR), a promising voice-based biometric recognition technique. In this work, we propose SLMIA-SR, the first membership inference attack tailored to SR. In contrast to conventional example-level attack, our attack features speaker-level membership inference, i.e., determining if any voices of a given speaker, either the same as or different from the given inference voices, have been involved in the training of a model. It is particularly useful and practical since the training and inference voices are usually distinct, and it is also meaningful considering the open-set nature of SR, namely, the recognition speakers were often not present in the training data. We utilize intra-similarity and inter-dissimilarity, two training objectives of SR, to characterize the differences between training and non-training speakers and quantify them with two groups of features driven by carefully-established feature engineering to mount the attack. To improve the generalizability of our attack, we propose a novel mixing ratio training strategy to train attack models. To enhance the attack performance, we introduce voice chunk splitting to cope with the limited number of inference voices and propose to train attack models dependent on the number of inference voices. Our attack is versatile and can work in both white-box and black-box scenarios. Additionally, we propose two novel techniques to reduce the number of black-box queries while maintaining the attack performance. Extensive experiments demonstrate the effectiveness of SLMIA-SR.



## **18. RetouchUAA: Unconstrained Adversarial Attack via Image Retouching**

cs.CV

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.16478v1) [paper-pdf](http://arxiv.org/pdf/2311.16478v1)

**Authors**: Mengda Xie, Yiling He, Meie Fang

**Abstract**: Deep Neural Networks (DNNs) are susceptible to adversarial examples. Conventional attacks generate controlled noise-like perturbations that fail to reflect real-world scenarios and hard to interpretable. In contrast, recent unconstrained attacks mimic natural image transformations occurring in the real world for perceptible but inconspicuous attacks, yet compromise realism due to neglect of image post-processing and uncontrolled attack direction. In this paper, we propose RetouchUAA, an unconstrained attack that exploits a real-life perturbation: image retouching styles, highlighting its potential threat to DNNs. Compared to existing attacks, RetouchUAA offers several notable advantages. Firstly, RetouchUAA excels in generating interpretable and realistic perturbations through two key designs: the image retouching attack framework and the retouching style guidance module. The former custom-designed human-interpretability retouching framework for adversarial attack by linearizing images while modelling the local processing and retouching decision-making in human retouching behaviour, provides an explicit and reasonable pipeline for understanding the robustness of DNNs against retouching. The latter guides the adversarial image towards standard retouching styles, thereby ensuring its realism. Secondly, attributed to the design of the retouching decision regularization and the persistent attack strategy, RetouchUAA also exhibits outstanding attack capability and defense robustness, posing a heavy threat to DNNs. Experiments on ImageNet and Place365 reveal that RetouchUAA achieves nearly 100\% white-box attack success against three DNNs, while achieving a better trade-off between image naturalness, transferability and defense robustness than baseline attacks.



## **19. Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information**

cs.CL

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.11509v2) [paper-pdf](http://arxiv.org/pdf/2311.11509v2)

**Authors**: Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan

**Abstract**: In recent years, Large Language Models (LLM) have emerged as pivotal tools in various applications. However, these models are susceptible to adversarial prompt attacks, where attackers can carefully curate input strings that lead to undesirable outputs. The inherent vulnerability of LLMs stems from their input-output mechanisms, especially when presented with intensely out-of-distribution (OOD) inputs. This paper proposes a token-level detection method to identify adversarial prompts, leveraging the LLM's capability to predict the next token's probability. We measure the degree of the model's perplexity and incorporate neighboring token information to encourage the detection of contiguous adversarial prompt sequences. As a result, we propose two methods: one that identifies each token as either being part of an adversarial prompt or not, and another that estimates the probability of each token being part of an adversarial prompt.



## **20. Instruct2Attack: Language-Guided Semantic Adversarial Attacks**

cs.CV

under submission, code coming soon

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15551v1) [paper-pdf](http://arxiv.org/pdf/2311.15551v1)

**Authors**: Jiang Liu, Chen Wei, Yuxiang Guo, Heng Yu, Alan Yuille, Soheil Feizi, Chun Pong Lau, Rama Chellappa

**Abstract**: We propose Instruct2Attack (I2A), a language-guided semantic attack that generates semantically meaningful perturbations according to free-form language instructions. We make use of state-of-the-art latent diffusion models, where we adversarially guide the reverse diffusion process to search for an adversarial latent code conditioned on the input image and text instruction. Compared to existing noise-based and semantic attacks, I2A generates more natural and diverse adversarial examples while providing better controllability and interpretability. We further automate the attack process with GPT-4 to generate diverse image-specific text instructions. We show that I2A can successfully break state-of-the-art deep neural networks even under strong adversarial defenses, and demonstrate great transferability among a variety of network architectures.



## **21. Confidence Is All You Need for MI Attacks**

cs.LG

2 pages, 1 figure

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.15373v1) [paper-pdf](http://arxiv.org/pdf/2311.15373v1)

**Authors**: Abhishek Sinha, Himanshi Tibrewal, Mansi Gupta, Nikhar Waghela, Shivank Garg

**Abstract**: In this evolving era of machine learning security, membership inference attacks have emerged as a potent threat to the confidentiality of sensitive data. In this attack, adversaries aim to determine whether a particular point was used during the training of a target model. This paper proposes a new method to gauge a data point's membership in a model's training set. Instead of correlating loss with membership, as is traditionally done, we have leveraged the fact that training examples generally exhibit higher confidence values when classified into their actual class. During training, the model is essentially being 'fit' to the training data and might face particular difficulties in generalization to unseen data. This asymmetry leads to the model achieving higher confidence on the training data as it exploits the specific patterns and noise present in the training data. Our proposed approach leverages the confidence values generated by the machine learning model. These confidence values provide a probabilistic measure of the model's certainty in its predictions and can further be used to infer the membership of a given data point. Additionally, we also introduce another variant of our method that allows us to carry out this attack without knowing the ground truth(true class) of a given data point, thus offering an edge over existing label-dependent attack methods.



## **22. Adversarial Purification of Information Masking**

cs.CV

**SubmitDate**: 2023-11-26    [abs](http://arxiv.org/abs/2311.15339v1) [paper-pdf](http://arxiv.org/pdf/2311.15339v1)

**Authors**: Sitong Liu, Zhichao Lian, Shuangquan Zhang, Liang Xiao

**Abstract**: Adversarial attacks meticulously generate minuscule, imperceptible perturbations to images to deceive neural networks. Counteracting these, adversarial purification methods seek to transform adversarial input samples into clean output images to defend against adversarial attacks. Nonetheless, extent generative models fail to effectively eliminate adversarial perturbations, yielding less-than-ideal purification results. We emphasize the potential threat of residual adversarial perturbations to target models, quantitatively establishing a relationship between perturbation scale and attack capability. Notably, the residual perturbations on the purified image primarily stem from the same-position patch and similar patches of the adversarial sample. We propose a novel adversarial purification approach named Information Mask Purification (IMPure), aims to extensively eliminate adversarial perturbations. To obtain an adversarial sample, we first mask part of the patches information, then reconstruct the patches to resist adversarial perturbations from the patches. We reconstruct all patches in parallel to obtain a cohesive image. Then, in order to protect the purified samples against potential similar regional perturbations, we simulate this risk by randomly mixing the purified samples with the input samples before inputting them into the feature extraction network. Finally, we establish a combined constraint of pixel loss and perceptual loss to augment the model's reconstruction adaptability. Extensive experiments on the ImageNet dataset with three classifier models demonstrate that our approach achieves state-of-the-art results against nine adversarial attack methods. Implementation code and pre-trained weights can be accessed at \textcolor{blue}{https://github.com/NoWindButRain/IMPure}.



## **23. Effective Backdoor Mitigation Depends on the Pre-training Objective**

cs.LG

Accepted for oral presentation at BUGS workshop @ NeurIPS 2023  (https://neurips2023-bugs.github.io/)

**SubmitDate**: 2023-11-25    [abs](http://arxiv.org/abs/2311.14948v1) [paper-pdf](http://arxiv.org/pdf/2311.14948v1)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for pre-training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in these models such as CleanCLIP which is the current state-of-the-art approach.   In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training.   We observe that stronger pre-training objectives correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP is ineffective when stronger pre-training objectives are used, even with extensive hyperparameter tuning.   Our findings underscore critical considerations for ML practitioners who pre-train models using large-scale web-curated data and are concerned about potential backdoor threats. Notably, our results suggest that simpler pre-training objectives are more amenable to effective backdoor removal. This insight is pivotal for practitioners seeking to balance the trade-offs between using stronger pre-training objectives and security against backdoor attacks.



## **24. Robust Graph Neural Networks via Unbiased Aggregation**

cs.LG

**SubmitDate**: 2023-11-25    [abs](http://arxiv.org/abs/2311.14934v1) [paper-pdf](http://arxiv.org/pdf/2311.14934v1)

**Authors**: Ruiqi Feng, Zhichao Hou, Tyler Derr, Xiaorui Liu

**Abstract**: The adversarial robustness of Graph Neural Networks (GNNs) has been questioned due to the false sense of security uncovered by strong adaptive attacks despite the existence of numerous defenses. In this work, we delve into the robustness analysis of representative robust GNNs and provide a unified robust estimation point of view to understand their robustness and limitations. Our novel analysis of estimation bias motivates the design of a robust and unbiased graph signal estimator. We then develop an efficient Quasi-Newton iterative reweighted least squares algorithm to solve the estimation problem, which unfolds as robust unbiased aggregation layers in GNNs with a theoretical convergence guarantee. Our comprehensive experiments confirm the strong robustness of our proposed model, and the ablation study provides a deep understanding of its advantages.



## **25. Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles**

cs.HC

10 pages, 16 tables, 5 figures, IEEE BigData 2023 (Workshops)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14876v1) [paper-pdf](http://arxiv.org/pdf/2311.14876v1)

**Authors**: Sonali Singh, Faranak Abri, Akbar Siami Namin

**Abstract**: With the recent advent of Large Language Models (LLMs), such as ChatGPT from OpenAI, BARD from Google, Llama2 from Meta, and Claude from Anthropic AI, gain widespread use, ensuring their security and robustness is critical. The widespread use of these language models heavily relies on their reliability and proper usage of this fascinating technology. It is crucial to thoroughly test these models to not only ensure its quality but also possible misuses of such models by potential adversaries for illegal activities such as hacking. This paper presents a novel study focusing on exploitation of such large language models against deceptive interactions. More specifically, the paper leverages widespread and borrows well-known techniques in deception theory to investigate whether these models are susceptible to deceitful interactions.   This research aims not only to highlight these risks but also to pave the way for robust countermeasures that enhance the security and integrity of language models in the face of sophisticated social engineering tactics. Through systematic experiments and analysis, we assess their performance in these critical security domains. Our results demonstrate a significant finding in that these large language models are susceptible to deception and social engineering attacks.



## **26. Adversarial Machine Learning in Latent Representations of Neural Networks**

cs.LG

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2309.17401v2) [paper-pdf](http://arxiv.org/pdf/2309.17401v2)

**Authors**: Milin Zhang, Mohammad Abdi, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge the resilience of distributed DNNs to adversarial action still remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and introduce two new measurements for distortion and robustness. Our theoretical findings indicate that (i) assuming the same level of information distortion, latent features are always more robust than input representations; (ii) the adversarial robustness is jointly determined by the feature dimension and the generalization capability of the DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN architectures, 6 different approaches for distributed DNN and 10 different adversarial attacks to the ImageNet-1K dataset. Our experimental results support our theoretical findings by showing that the compressed latent representations can reduce the success rate of adversarial attacks by 88% in the best case and by 57% on the average compared to attacks to the input space.



## **27. Tamper-Evident Pairing**

cs.CR

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14790v1) [paper-pdf](http://arxiv.org/pdf/2311.14790v1)

**Authors**: Aleksandar Manev

**Abstract**: Establishing a secure connection between wireless devices has become significantly important with the increasing number of Wi-Fi products coming to the market. In order to provide an easy and secure pairing standard, the Wi-Fi Alliance has designed the Wi-Fi Protected Setup. Push-Button Configuration (PBC) is part of this standard and is especially useful for pairing devices with physical limitations. However, PBC is proven to be vulnerable to man-in-the-middle (MITM) attacks. Tamper-Evident Pairing (TEP) is an improvement of the PBC standard, which aims to fix the MITM vulnerability without interfering the useful properties of PBC. It relies on the Tamper-Evident Announcement (TEA), which guarantees that an adversary can neither tamper a transmitted message without being detected, nor hide the fact that the message has been sent. The security properties of TEP were proven manually by its authors and tested with the Uppaal and Spin model checkers. During the Uppaal model checking, no vulnerabilities were found. However, the Spin model revealed a case, in which the TEP's security is not guaranteed. In this paper, we first provide a comprehensive overview of the TEP protocol, including all information needed to understand how it works. Furthermore, we summarize the security checks performed on it, give the circumstances, under which it is no longer resistant to MITM attacks and explain the reasons why they could not be revealed with the first model. Nevertheless, future work is required to gain full certainty of the TEP's security before applying it in the industry.



## **28. Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers**

cs.LG

In ICML 2021. Fixed typos in Eq. (3) and Eq. (4)

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2103.01208v3) [paper-pdf](http://arxiv.org/pdf/2103.01208v3)

**Authors**: Francesco Croce, Matthias Hein

**Abstract**: We show that when taking into account also the image domain $[0,1]^d$, established $l_1$-projected gradient descent (PGD) attacks are suboptimal as they do not consider that the effective threat model is the intersection of the $l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest descent step for this effective threat model and show that the exact projection onto this set is computationally feasible and yields better performance. Moreover, we propose an adaptive form of PGD which is highly effective even with a small budget of iterations. Our resulting $l_1$-APGD is a strong white-box attack showing that prior works overestimated their $l_1$-robustness. Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA $l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which reliably assesses adversarial robustness for the threat model of $l_1$-ball intersected with $[0,1]^d$.



## **29. Trainwreck: A damaging adversarial attack on image classifiers**

cs.CV

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14772v1) [paper-pdf](http://arxiv.org/pdf/2311.14772v1)

**Authors**: Jan Zahálka

**Abstract**: Adversarial attacks are an important security concern for computer vision (CV), as they enable malicious attackers to reliably manipulate CV models. Existing attacks aim to elicit an output desired by the attacker, but keep the model fully intact on clean data. With CV models becoming increasingly valuable assets in applied practice, a new attack vector is emerging: disrupting the models as a form of economic sabotage. This paper opens up the exploration of damaging adversarial attacks (DAAs) that seek to damage the target model and maximize the total cost incurred by the damage. As a pioneer DAA, this paper proposes Trainwreck, a train-time attack that poisons the training data of image classifiers to degrade their performance. Trainwreck conflates the data of similar classes using stealthy ($\epsilon \leq 8/255$) class-pair universal perturbations computed using a surrogate model. Trainwreck is a black-box, transferable attack: it requires no knowledge of the target model's architecture, and a single poisoned dataset degrades the performance of any model trained on it. The experimental evaluation on CIFAR-10 and CIFAR-100 demonstrates that Trainwreck is indeed an effective attack across various model architectures including EfficientNetV2, ResNeXt-101, and a finetuned ViT-L-16. The strength of the attack can be customized by the poison rate parameter. Finally, data redundancy with file hashing and/or pixel difference are identified as a reliable defense technique against Trainwreck or similar DAAs. The code is available at https://github.com/JanZahalka/trainwreck.



## **30. Universal Jailbreak Backdoors from Poisoned Human Feedback**

cs.AI

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14455v1) [paper-pdf](http://arxiv.org/pdf/2311.14455v1)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.



## **31. Segment (Almost) Nothing: Prompt-Agnostic Adversarial Attacks on Segmentation Models**

cs.CV

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14450v1) [paper-pdf](http://arxiv.org/pdf/2311.14450v1)

**Authors**: Francesco Croce, Matthias Hein

**Abstract**: General purpose segmentation models are able to generate (semantic) segmentation masks from a variety of prompts, including visual (points, boxed, etc.) and textual (object names) ones. In particular, input images are pre-processed by an image encoder to obtain embedding vectors which are later used for mask predictions. Existing adversarial attacks target the end-to-end tasks, i.e. aim at altering the segmentation mask predicted for a specific image-prompt pair. However, this requires running an individual attack for each new prompt for the same image. We propose instead to generate prompt-agnostic adversarial attacks by maximizing the $\ell_2$-distance, in the latent space, between the embedding of the original and perturbed images. Since the encoding process only depends on the image, distorted image representations will cause perturbations in the segmentation masks for a variety of prompts. We show that even imperceptible $\ell_\infty$-bounded perturbations of radius $\epsilon=1/255$ are often sufficient to drastically modify the masks predicted with point, box and text prompts by recently proposed foundation models for segmentation. Moreover, we explore the possibility of creating universal, i.e. non image-specific, attacks which can be readily applied to any input without further computational cost.



## **32. Federated Transformed Learning for a Circular, Secure, and Tiny AI**

cs.NI

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.14371v1) [paper-pdf](http://arxiv.org/pdf/2311.14371v1)

**Authors**: Weisi Guo, Schyler Sun, Bin Li, Sam Blakeman

**Abstract**: Deep Learning (DL) is penetrating into a diverse range of mass mobility, smart living, and industrial applications, rapidly transforming the way we live and work. DL is at the heart of many AI implementations. A key set of challenges is to produce AI modules that are: (1) "circular" - can solve new tasks without forgetting how to solve previous ones, (2) "secure" - have immunity to adversarial data attacks, and (3) "tiny" - implementable in low power low cost embedded hardware. Clearly it is difficult to achieve all three aspects on a single horizontal layer of platforms, as the techniques require transformed deep representations that incur different computation and communication requirements. Here we set out the vision to achieve transformed DL representations across a 5G and Beyond networked architecture. We first detail the cross-sectoral motivations for each challenge area, before demonstrating recent advances in DL research that can achieve circular, secure, and tiny AI (CST-AI). Recognising the conflicting demand of each transformed deep representation, we federate their deep learning transformations and functionalities across the network to achieve connected run-time capabilities.



## **33. BrainWash: A Poisoning Attack to Forget in Continual Learning**

cs.LG

**SubmitDate**: 2023-11-24    [abs](http://arxiv.org/abs/2311.11995v3) [paper-pdf](http://arxiv.org/pdf/2311.11995v3)

**Authors**: Ali Abbasi, Parsa Nooralinejad, Hamed Pirsiavash, Soheil Kolouri

**Abstract**: Continual learning has gained substantial attention within the deep learning community, offering promising solutions to the challenging problem of sequential learning. Yet, a largely unexplored facet of this paradigm is its susceptibility to adversarial attacks, especially with the aim of inducing forgetting. In this paper, we introduce "BrainWash," a novel data poisoning method tailored to impose forgetting on a continual learner. By adding the BrainWash noise to a variety of baselines, we demonstrate how a trained continual learner can be induced to forget its previously learned tasks catastrophically, even when using these continual learning baselines. An important feature of our approach is that the attacker requires no access to previous tasks' data and is armed merely with the model's current parameters and the data belonging to the most recent task. Our extensive experiments highlight the efficacy of BrainWash, showcasing degradation in performance across various regularization-based continual learning methods.



## **34. When Side-Channel Attacks Break the Black-Box Property of Embedded Artificial Intelligence**

cs.AI

**SubmitDate**: 2023-11-23    [abs](http://arxiv.org/abs/2311.14005v1) [paper-pdf](http://arxiv.org/pdf/2311.14005v1)

**Authors**: Benoit Coqueret, Mathieu Carbone, Olivier Sentieys, Gabriel Zaid

**Abstract**: Artificial intelligence, and specifically deep neural networks (DNNs), has rapidly emerged in the past decade as the standard for several tasks from specific advertising to object detection. The performance offered has led DNN algorithms to become a part of critical embedded systems, requiring both efficiency and reliability. In particular, DNNs are subject to malicious examples designed in a way to fool the network while being undetectable to the human observer: the adversarial examples. While previous studies propose frameworks to implement such attacks in black box settings, those often rely on the hypothesis that the attacker has access to the logits of the neural network, breaking the assumption of the traditional black box. In this paper, we investigate a real black box scenario where the attacker has no access to the logits. In particular, we propose an architecture-agnostic attack which solve this constraint by extracting the logits. Our method combines hardware and software attacks, by performing a side-channel attack that exploits electromagnetic leakages to extract the logits for a given input, allowing an attacker to estimate the gradients and produce state-of-the-art adversarial examples to fool the targeted neural network. Through this example of adversarial attack, we demonstrate the effectiveness of logits extraction using side-channel as a first step for more general attack frameworks requiring either the logits or the confidence scores.



## **35. An Extensive Study on Adversarial Attack against Pre-trained Models of Code**

cs.CR

Accepted to ESEC/FSE 2023

**SubmitDate**: 2023-11-23    [abs](http://arxiv.org/abs/2311.07553v2) [paper-pdf](http://arxiv.org/pdf/2311.07553v2)

**Authors**: Xiaohu Du, Ming Wen, Zichao Wei, Shangwen Wang, Hai Jin

**Abstract**: Transformer-based pre-trained models of code (PTMC) have been widely utilized and have achieved state-of-the-art performance in many mission-critical applications. However, they can be vulnerable to adversarial attacks through identifier substitution or coding style transformation, which can significantly degrade accuracy and may further incur security concerns. Although several approaches have been proposed to generate adversarial examples for PTMC, the effectiveness and efficiency of such approaches, especially on different code intelligence tasks, has not been well understood. To bridge this gap, this study systematically analyzes five state-of-the-art adversarial attack approaches from three perspectives: effectiveness, efficiency, and the quality of generated examples. The results show that none of the five approaches balances all these perspectives. Particularly, approaches with a high attack success rate tend to be time-consuming; the adversarial code they generate often lack naturalness, and vice versa. To address this limitation, we explore the impact of perturbing identifiers under different contexts and find that identifier substitution within for and if statements is the most effective. Based on these findings, we propose a new approach that prioritizes different types of statements for various tasks and further utilizes beam search to generate adversarial examples. Evaluation results show that it outperforms the state-of-the-art ALERT in terms of both effectiveness and efficiency while preserving the naturalness of the generated adversarial examples.



## **36. Adversarial defense based on distribution transfer**

cs.CR

27 pages

**SubmitDate**: 2023-11-23    [abs](http://arxiv.org/abs/2311.13841v1) [paper-pdf](http://arxiv.org/pdf/2311.13841v1)

**Authors**: Jiahao Chen, Diqun Yan, Li Dong

**Abstract**: The presence of adversarial examples poses a significant threat to deep learning models and their applications. Existing defense methods provide certain resilience against adversarial examples, but often suffer from decreased accuracy and generalization performance, making it challenging to achieve a trade-off between robustness and generalization. To address this, our paper interprets the adversarial example problem from the perspective of sample distribution and proposes a defense method based on distribution shift, leveraging the distribution transfer capability of a diffusion model for adversarial defense. The core idea is to exploit the discrepancy between normal and adversarial sample distributions to achieve adversarial defense using a pretrained diffusion model. Specifically, an adversarial sample undergoes a forward diffusion process, moving away from the source distribution, followed by a reverse process guided by the protected model (victim model) output to map it back to the normal distribution. Experimental evaluations on CIFAR10 and ImageNet30 datasets are conducted, comparing with adversarial training and input preprocessing methods. For infinite-norm attacks with 8/255 perturbation, accuracy rates of 78.1% and 83.5% are achieved, respectively. For 2-norm attacks with 128/255 perturbation, accuracy rates are 74.3% and 82.5%. Additional experiments considering perturbation amplitude, diffusion iterations, and adaptive attacks also validate the effectiveness of the proposed method. Results demonstrate that even when the attacker has knowledge of the defense, the proposed distribution-based method effectively withstands adversarial examples. It fills the gaps of traditional approaches, restoring high-quality original samples and showcasing superior performance in model robustness and generalization.



## **37. Security and Privacy Challenges in Deep Learning Models**

cs.CR

**SubmitDate**: 2023-11-23    [abs](http://arxiv.org/abs/2311.13744v1) [paper-pdf](http://arxiv.org/pdf/2311.13744v1)

**Authors**: Gopichandh Golla

**Abstract**: These days, deep learning models have achieved great success in multiple fields, from autonomous driving to medical diagnosis. These models have expanded the abilities of artificial intelligence by offering great solutions to complex problems that were very difficult to solve earlier. In spite of their unseen success in various, it has been identified, through research conducted, that deep learning models can be subjected to various attacks that compromise model security and data privacy of the Deep Neural Network models. Deep learning models can be subjected to various attacks at different stages of their lifecycle. During the testing phase, attackers can exploit vulnerabilities through different kinds of attacks such as Model Extraction Attacks, Model Inversion attacks, and Adversarial attacks. Model Extraction Attacks are aimed at reverse-engineering a trained deep learning model, with the primary objective of revealing its architecture and parameters. Model inversion attacks aim to compromise the privacy of the data used in the Deep learning model. These attacks are done to compromise the confidentiality of the model by going through the sensitive training data from the model's predictions. By analyzing the model's responses, attackers aim to reconstruct sensitive information. In this way, the model's data privacy is compromised. Adversarial attacks, mainly employed on computer vision models, are made to corrupt models into confidently making incorrect predictions through malicious testing data. These attacks subtly alter the input data, making it look normal but misleading deep learning models to make incorrect decisions. Such attacks can happen during both the model's evaluation and training phases. Data Poisoning Attacks add harmful data to the training set, disrupting the learning process and reducing the reliability of the deep learning mode.



## **38. Panda or not Panda? Understanding Adversarial Attacks with Interactive Visualization**

cs.HC

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13656v1) [paper-pdf](http://arxiv.org/pdf/2311.13656v1)

**Authors**: Yuzhe You, Jarvis Tse, Jian Zhao

**Abstract**: Adversarial machine learning (AML) studies attacks that can fool machine learning algorithms into generating incorrect outcomes as well as the defenses against worst-case attacks to strengthen model robustness. Specifically for image classification, it is challenging to understand adversarial attacks due to their use of subtle perturbations that are not human-interpretable, as well as the variability of attack impacts influenced by diverse methodologies, instance differences, and model architectures. Through a design study with AML learners and teachers, we introduce AdvEx, a multi-level interactive visualization system that comprehensively presents the properties and impacts of evasion attacks on different image classifiers for novice AML learners. We quantitatively and qualitatively assessed AdvEx in a two-part evaluation including user studies and expert interviews. Our results show that AdvEx is not only highly effective as a visualization tool for understanding AML mechanisms, but also provides an engaging and enjoyable learning experience, thus demonstrating its overall benefits for AML learners.



## **39. Adversarial Backdoor Attack by Naturalistic Data Poisoning on Trajectory Prediction in Autonomous Driving**

cs.CV

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2306.15755v2) [paper-pdf](http://arxiv.org/pdf/2306.15755v2)

**Authors**: Mozhgan Pourkeshavarz, Mohammad Sabokrou, Amir Rasouli

**Abstract**: In autonomous driving, behavior prediction is fundamental for safe motion planning, hence the security and robustness of prediction models against adversarial attacks are of paramount importance. We propose a novel adversarial backdoor attack against trajectory prediction models as a means of studying their potential vulnerabilities. Our attack affects the victim at training time via naturalistic, hence stealthy, poisoned samples crafted using a novel two-step approach. First, the triggers are crafted by perturbing the trajectory of attacking vehicle and then disguised by transforming the scene using a bi-level optimization technique. The proposed attack does not depend on a particular model architecture and operates in a black-box manner, thus can be effective without any knowledge of the victim model. We conduct extensive empirical studies using state-of-the-art prediction models on two benchmark datasets using metrics customized for trajectory prediction. We show that the proposed attack is highly effective, as it can significantly hinder the performance of prediction models, unnoticeable by the victims, and efficient as it forces the victim to generate malicious behavior even under constrained conditions. Via ablative studies, we analyze the impact of different attack design choices followed by an evaluation of existing defence mechanisms against the proposed attack.



## **40. Transfer Attacks and Defenses for Large Language Models on Coding Tasks**

cs.LG

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13445v1) [paper-pdf](http://arxiv.org/pdf/2311.13445v1)

**Authors**: Chi Zhang, Zifan Wang, Ravi Mangal, Matt Fredrikson, Limin Jia, Corina Pasareanu

**Abstract**: Modern large language models (LLMs), such as ChatGPT, have demonstrated impressive capabilities for coding tasks including writing and reasoning about code. They improve upon previous neural network models of code, such as code2seq or seq2seq, that already demonstrated competitive results when performing tasks such as code summarization and identifying code vulnerabilities. However, these previous code models were shown vulnerable to adversarial examples, i.e. small syntactic perturbations that do not change the program's semantics, such as the inclusion of "dead code" through false conditions or the addition of inconsequential print statements, designed to "fool" the models. LLMs can also be vulnerable to the same adversarial perturbations but a detailed study on this concern has been lacking so far. In this paper we aim to investigate the effect of adversarial perturbations on coding tasks with LLMs. In particular, we study the transferability of adversarial examples, generated through white-box attacks on smaller code models, to LLMs. Furthermore, to make the LLMs more robust against such adversaries without incurring the cost of retraining, we propose prompt-based defenses that involve modifying the prompt to include additional information such as examples of adversarially perturbed code and explicit instructions for reversing adversarial perturbations. Our experiments show that adversarial examples obtained with a smaller code model are indeed transferable, weakening the LLMs' performance. The proposed defenses show promise in improving the model's resilience, paving the way to more robust defensive solutions for LLMs in code-related applications.



## **41. From Principle to Practice: Vertical Data Minimization for Machine Learning**

cs.LG

Accepted at IEEE S&P 2024

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.10500v2) [paper-pdf](http://arxiv.org/pdf/2311.10500v2)

**Authors**: Robin Staab, Nikola Jovanović, Mislav Balunović, Martin Vechev

**Abstract**: Aiming to train and deploy predictive models, organizations collect large amounts of detailed client data, risking the exposure of private information in the event of a breach. To mitigate this, policymakers increasingly demand compliance with the data minimization (DM) principle, restricting data collection to only that data which is relevant and necessary for the task. Despite regulatory pressure, the problem of deploying machine learning models that obey DM has so far received little attention. In this work, we address this challenge in a comprehensive manner. We propose a novel vertical DM (vDM) workflow based on data generalization, which by design ensures that no full-resolution client data is collected during training and deployment of models, benefiting client privacy by reducing the attack surface in case of a breach. We formalize and study the corresponding problem of finding generalizations that both maximize data utility and minimize empirical privacy risk, which we quantify by introducing a diverse set of policy-aligned adversarial scenarios. Finally, we propose a range of baseline vDM algorithms, as well as Privacy-aware Tree (PAT), an especially effective vDM algorithm that outperforms all baselines across several settings. We plan to release our code as a publicly available library, helping advance the standardization of DM for machine learning. Overall, we believe our work can help lay the foundation for further exploration and adoption of DM principles in real-world applications.



## **42. Hard Label Black Box Node Injection Attack on Graph Neural Networks**

cs.LG

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13244v1) [paper-pdf](http://arxiv.org/pdf/2311.13244v1)

**Authors**: Yu Zhou, Zihao Dong, Guofeng Zhang, Jingchen Tang

**Abstract**: While graph neural networks have achieved state-of-the-art performances in many real-world tasks including graph classification and node classification, recent works have demonstrated they are also extremely vulnerable to adversarial attacks. Most previous works have focused on attacking node classification networks under impractical white-box scenarios. In this work, we will propose a non-targeted Hard Label Black Box Node Injection Attack on Graph Neural Networks, which to the best of our knowledge, is the first of its kind. Under this setting, more real world tasks can be studied because our attack assumes no prior knowledge about (1): the model architecture of the GNN we are attacking; (2): the model's gradients; (3): the output logits of the target GNN model. Our attack is based on an existing edge perturbation attack, from which we restrict the optimization process to formulate a node injection attack. In the work, we will evaluate the performance of the attack using three datasets, COIL-DEL, IMDB-BINARY, and NCI1.



## **43. A Survey of Adversarial CAPTCHAs on its History, Classification and Generation**

cs.CR

Submitted to ACM Computing Surveys (Under Review)

**SubmitDate**: 2023-11-22    [abs](http://arxiv.org/abs/2311.13233v1) [paper-pdf](http://arxiv.org/pdf/2311.13233v1)

**Authors**: Zisheng Xu, Qiao Yan, F. Richard Yu, Victor C. M. Leung

**Abstract**: Completely Automated Public Turing test to tell Computers and Humans Apart, short for CAPTCHA, is an essential and relatively easy way to defend against malicious attacks implemented by bots. The security and usability trade-off limits the use of massive geometric transformations to interfere deep model recognition and deep models even outperformed humans in complex CAPTCHAs. The discovery of adversarial examples provides an ideal solution to the security and usability trade-off by integrating adversarial examples and CAPTCHAs to generate adversarial CAPTCHAs that can fool the deep models. In this paper, we extend the definition of adversarial CAPTCHAs and propose a classification method for adversarial CAPTCHAs. Then we systematically review some commonly used methods to generate adversarial examples and methods that are successfully used to generate adversarial CAPTCHAs. Also, we analyze some defense methods that can be used to defend adversarial CAPTCHAs, indicating potential threats to adversarial CAPTCHAs. Finally, we discuss some possible future research directions for adversarial CAPTCHAs at the end of this paper.



## **44. HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks**

cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2309.08549v3) [paper-pdf](http://arxiv.org/pdf/2309.08549v3)

**Authors**: Minh-Hao Van, Alycia N. Carey, Xintao Wu

**Abstract**: While numerous defense methods have been proposed to prohibit potential poisoning attacks from untrusted data sources, most research works only defend against specific attacks, which leaves many avenues for an adversary to exploit. In this work, we propose an efficient and robust training approach to defend against data poisoning attacks based on influence functions, named Healthy Influential-Noise based Training. Using influence functions, we craft healthy noise that helps to harden the classification model against poisoning attacks without significantly affecting the generalization ability on test data. In addition, our method can perform effectively when only a subset of the training data is modified, instead of the current method of adding noise to all examples that has been used in several previous works. We conduct comprehensive evaluations over two image datasets with state-of-the-art poisoning attacks under different realistic attack scenarios. Our empirical results show that HINT can efficiently protect deep learning models against the effect of both untargeted and targeted poisoning attacks.



## **45. Epsilon*: Privacy Metric for Machine Learning Models**

cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2307.11280v2) [paper-pdf](http://arxiv.org/pdf/2307.11280v2)

**Authors**: Diana M. Negoescu, Humberto Gonzalez, Saad Eddin Al Orjany, Jilei Yang, Yuliia Lut, Rahul Tandra, Xiaowen Zhang, Xinyi Zheng, Zach Douglas, Vidita Nolkha, Parvez Ahammad, Gennady Samorodnitsky

**Abstract**: We introduce Epsilon*, a new privacy metric for measuring the privacy risk of a single model instance prior to, during, or after deployment of privacy mitigation strategies. The metric requires only black-box access to model predictions, does not require training data re-sampling or model re-training, and can be used to measure the privacy risk of models not trained with differential privacy. Epsilon* is a function of true positive and false positive rates in a hypothesis test used by an adversary in a membership inference attack. We distinguish between quantifying the privacy loss of a trained model instance, which we refer to as empirical privacy, and quantifying the privacy loss of the training mechanism which produces this model instance. Existing approaches in the privacy auditing literature provide lower bounds for the latter, while our metric provides an empirical lower bound for the former by relying on an (${\epsilon}$, ${\delta}$)-type of quantification of the privacy of the trained model instance. We establish a relationship between these lower bounds and show how to implement Epsilon* to avoid numerical and noise amplification instability. We further show in experiments on benchmark public data sets that Epsilon* is sensitive to privacy risk mitigation by training with differential privacy (DP), where the value of Epsilon* is reduced by up to 800% compared to the Epsilon* values of non-DP trained baseline models. This metric allows privacy auditors to be independent of model owners, and enables visualizing the privacy-utility landscape to make informed decisions regarding the trade-offs between model privacy and utility.



## **46. Is your vote truly secret? Ballot Secrecy iff Ballot Independence: Proving necessary conditions and analysing case studies**

cs.CR

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12977v1) [paper-pdf](http://arxiv.org/pdf/2311.12977v1)

**Authors**: Aida Manzano Kharman, Ben Smyth, Freddie Page

**Abstract**: We formalise definitions of ballot secrecy and ballot independence by Smyth, JCS'21 as indistinguishability games in the computational model of security. These definitions improve upon Smyth, draft '21 to consider a wider class of voting systems. Both Smyth, JCS'21 and Smyth, draft '21 improve on earlier works by considering a more realistic adversary model wherein they have access to the ballot collection. We prove that ballot secrecy implies ballot independence. We say ballot independence holds if a system has non-malleable ballots. We construct games for ballot secrecy and non-malleability and show that voting schemes with malleable ballots do not preserve ballot secrecy. We demonstrate that Helios does not satisfy our definition of ballot secrecy. Furthermore, the Python framework we constructed for our case study shows that if an attack exists against non-malleability, this attack can be used to break ballot secrecy.



## **47. Iris Presentation Attack: Assessing the Impact of Combining Vanadium Dioxide Films with Artificial Eyes**

cs.CV

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12773v1) [paper-pdf](http://arxiv.org/pdf/2311.12773v1)

**Authors**: Darshika Jauhari, Renu Sharma, Cunjian Chen, Nelson Sepulveda, Arun Ross

**Abstract**: Iris recognition systems, operating in the near infrared spectrum (NIR), have demonstrated vulnerability to presentation attacks, where an adversary uses artifacts such as cosmetic contact lenses, artificial eyes or printed iris images in order to circumvent the system. At the same time, a number of effective presentation attack detection (PAD) methods have been developed. These methods have demonstrated success in detecting artificial eyes (e.g., fake Van Dyke eyes) as presentation attacks. In this work, we seek to alter the optical characteristics of artificial eyes by affixing Vanadium Dioxide (VO2) films on their surface in various spatial configurations. VO2 films can be used to selectively transmit NIR light and can, therefore, be used to regulate the amount of NIR light from the object that is captured by the iris sensor. We study the impact of such images produced by the sensor on two state-of-the-art iris PA detection methods. We observe that the addition of VO2 films on the surface of artificial eyes can cause the PA detection methods to misclassify them as bonafide eyes in some cases. This represents a vulnerability that must be systematically analyzed and effectively addressed.



## **48. Attention Deficit is Ordered! Fooling Deformable Vision Transformers with Collaborative Adversarial Patches**

cs.CV

9 pages, 10 figures

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12914v1) [paper-pdf](http://arxiv.org/pdf/2311.12914v1)

**Authors**: Quazi Mishkatul Alam, Bilel Tarchoun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: The latest generation of transformer-based vision models have proven to be superior to Convolutional Neural Network (CNN)-based models across several vision tasks, largely attributed to their remarkable prowess in relation modeling. Deformable vision transformers significantly reduce the quadratic complexity of modeling attention by using sparse attention structures, enabling them to be used in larger scale applications such as multi-view vision systems. Recent work demonstrated adversarial attacks against transformers; we show that these attacks do not transfer to deformable transformers due to their sparse attention structure. Specifically, attention in deformable transformers is modeled using pointers to the most relevant other tokens. In this work, we contribute for the first time adversarial attacks that manipulate the attention of deformable transformers, distracting them to focus on irrelevant parts of the image. We also develop new collaborative attacks where a source patch manipulates attention to point to a target patch that adversarially attacks the system. In our experiments, we find that only 1% patched area of the input field can lead to 0% AP. We also show that the attacks provide substantial versatility to support different attacker scenarios because of their ability to redirect attention under the attacker control.



## **49. Attacking Motion Planners Using Adversarial Perception Errors**

cs.RO

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2311.12722v1) [paper-pdf](http://arxiv.org/pdf/2311.12722v1)

**Authors**: Jonathan Sadeghi, Nicholas A. Lord, John Redford, Romain Mueller

**Abstract**: Autonomous driving (AD) systems are often built and tested in a modular fashion, where the performance of different modules is measured using task-specific metrics. These metrics should be chosen so as to capture the downstream impact of each module and the performance of the system as a whole. For example, high perception quality should enable prediction and planning to be performed safely. Even though this is true in general, we show here that it is possible to construct planner inputs that score very highly on various perception quality metrics but still lead to planning failures. In an analogy to adversarial attacks on image classifiers, we call such inputs \textbf{adversarial perception errors} and show they can be systematically constructed using a simple boundary-attack algorithm. We demonstrate the effectiveness of this algorithm by finding attacks for two different black-box planners in several urban and highway driving scenarios using the CARLA simulator. Finally, we analyse the properties of these attacks and show that they are isolated in the input space of the planner, and discuss their implications for AD system deployment and testing.



## **50. Differentially Private Optimizers Can Learn Adversarially Robust Models**

cs.LG

**SubmitDate**: 2023-11-21    [abs](http://arxiv.org/abs/2211.08942v2) [paper-pdf](http://arxiv.org/pdf/2211.08942v2)

**Authors**: Yuan Zhang, Zhiqi Bu

**Abstract**: Machine learning models have shone in a variety of domains and attracted increasing attention from both the security and the privacy communities. One important yet worrying question is: Will training models under the differential privacy (DP) constraint have an unfavorable impact on their adversarial robustness? While previous works have postulated that privacy comes at the cost of worse robustness, we give the first theoretical analysis to show that DP models can indeed be robust and accurate, even sometimes more robust than their naturally-trained non-private counterparts. We observe three key factors that influence the privacy-robustness-accuracy tradeoff: (1) hyper-parameters for DP optimizers are critical; (2) pre-training on public data significantly mitigates the accuracy and robustness drop; (3) choice of DP optimizers makes a difference. With these factors set properly, we achieve 90\% natural accuracy, 72\% robust accuracy ($+9\%$ than the non-private model) under $l_2(0.5)$ attack, and 69\% robust accuracy ($+16\%$ than the non-private model) with pre-trained SimCLRv2 model under $l_\infty(4/255)$ attack on CIFAR10 with $\epsilon=2$. In fact, we show both theoretically and empirically that DP models are Pareto optimal on the accuracy-robustness tradeoff. Empirically, the robustness of DP models is consistently observed across various datasets and models. We believe our encouraging results are a significant step towards training models that are private as well as robust.



