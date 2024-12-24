# Latest Adversarial Attack Papers
**update at 2024-12-24 10:06:31**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

cs.CV

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.17038v1) [paper-pdf](http://arxiv.org/pdf/2412.17038v1)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Ziyi Liu

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.



## **2. Robustness of Large Language Models Against Adversarial Attacks**

cs.CL

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.17011v1) [paper-pdf](http://arxiv.org/pdf/2412.17011v1)

**Authors**: Yiyi Tao, Yixian Shen, Hang Zhang, Yanxin Shen, Lun Wang, Chuanqi Shi, Shaoshuai Du

**Abstract**: The increasing deployment of Large Language Models (LLMs) in various applications necessitates a rigorous evaluation of their robustness against adversarial attacks. In this paper, we present a comprehensive study on the robustness of GPT LLM family. We employ two distinct evaluation methods to assess their resilience. The first method introduce character-level text attack in input prompts, testing the models on three sentiment classification datasets: StanfordNLP/IMDB, Yelp Reviews, and SST-2. The second method involves using jailbreak prompts to challenge the safety mechanisms of the LLMs. Our experiments reveal significant variations in the robustness of these models, demonstrating their varying degrees of vulnerability to both character-level and semantic-level adversarial attacks. These findings underscore the necessity for improved adversarial training and enhanced safety mechanisms to bolster the robustness of LLMs.



## **3. Breaking Barriers in Physical-World Adversarial Examples: Improving Robustness and Transferability via Robust Feature**

cs.CV

Accepted by AAAI2025

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16958v1) [paper-pdf](http://arxiv.org/pdf/2412.16958v1)

**Authors**: Yichen Wang, Yuxuan Chou, Ziqi Zhou, Hangtao Zhang, Wei Wan, Shengshan Hu, Minghui Li

**Abstract**: As deep neural networks (DNNs) are widely applied in the physical world, many researches are focusing on physical-world adversarial examples (PAEs), which introduce perturbations to inputs and cause the model's incorrect outputs. However, existing PAEs face two challenges: unsatisfactory attack performance (i.e., poor transferability and insufficient robustness to environment conditions), and difficulty in balancing attack effectiveness with stealthiness, where better attack effectiveness often makes PAEs more perceptible.   In this paper, we explore a novel perturbation-based method to overcome the challenges. For the first challenge, we introduce a strategy Deceptive RF injection based on robust features (RFs) that are predictive, robust to perturbations, and consistent across different models. Specifically, it improves the transferability and robustness of PAEs by covering RFs of other classes onto the predictive features in clean images. For the second challenge, we introduce another strategy Adversarial Semantic Pattern Minimization, which removes most perturbations and retains only essential adversarial patterns in AEsBased on the two strategies, we design our method Robust Feature Coverage Attack (RFCoA), comprising Robust Feature Disentanglement and Adversarial Feature Fusion. In the first stage, we extract target class RFs in feature space. In the second stage, we use attention-based feature fusion to overlay these RFs onto predictive features of clean images and remove unnecessary perturbations. Experiments show our method's superior transferability, robustness, and stealthiness compared to existing state-of-the-art methods. Additionally, our method's effectiveness can extend to Large Vision-Language Models (LVLMs), indicating its potential applicability to more complex tasks.



## **4. NumbOD: A Spatial-Frequency Fusion Attack Against Object Detectors**

cs.CV

Accepted by AAAI 2025

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16955v1) [paper-pdf](http://arxiv.org/pdf/2412.16955v1)

**Authors**: Ziqi Zhou, Bowen Li, Yufei Song, Zhifei Yu, Shengshan Hu, Wei Wan, Leo Yu Zhang, Dezhong Yao, Hai Jin

**Abstract**: With the advancement of deep learning, object detectors (ODs) with various architectures have achieved significant success in complex scenarios like autonomous driving. Previous adversarial attacks against ODs have been focused on designing customized attacks targeting their specific structures (e.g., NMS and RPN), yielding some results but simultaneously constraining their scalability. Moreover, most efforts against ODs stem from image-level attacks originally designed for classification tasks, resulting in redundant computations and disturbances in object-irrelevant areas (e.g., background). Consequently, how to design a model-agnostic efficient attack to comprehensively evaluate the vulnerabilities of ODs remains challenging and unresolved. In this paper, we propose NumbOD, a brand-new spatial-frequency fusion attack against various ODs, aimed at disrupting object detection within images. We directly leverage the features output by the OD without relying on its internal structures to craft adversarial examples. Specifically, we first design a dual-track attack target selection strategy to select high-quality bounding boxes from OD outputs for targeting. Subsequently, we employ directional perturbations to shift and compress predicted boxes and change classification results to deceive ODs. Additionally, we focus on manipulating the high-frequency components of images to confuse ODs' attention on critical objects, thereby enhancing the attack efficiency. Our extensive experiments on nine ODs and two datasets show that NumbOD achieves powerful attack performance and high stealthiness.



## **5. Preventing Non-intrusive Load Monitoring Privacy Invasion: A Precise Adversarial Attack Scheme for Networked Smart Meters**

cs.CR

**SubmitDate**: 2024-12-22    [abs](http://arxiv.org/abs/2412.16893v1) [paper-pdf](http://arxiv.org/pdf/2412.16893v1)

**Authors**: Jialing He, Jiacheng Wang, Ning Wang, Shangwei Guo, Liehuang Zhu, Dusit Niyato, Tao Xiang

**Abstract**: Smart grid, through networked smart meters employing the non-intrusive load monitoring (NILM) technique, can considerably discern the usage patterns of residential appliances. However, this technique also incurs privacy leakage. To address this issue, we propose an innovative scheme based on adversarial attack in this paper. The scheme effectively prevents NILM models from violating appliance-level privacy, while also ensuring accurate billing calculation for users. To achieve this objective, we overcome two primary challenges. First, as NILM models fall under the category of time-series regression models, direct application of traditional adversarial attacks designed for classification tasks is not feasible. To tackle this issue, we formulate a novel adversarial attack problem tailored specifically for NILM and providing a theoretical foundation for utilizing the Jacobian of the NILM model to generate imperceptible perturbations. Leveraging the Jacobian, our scheme can produce perturbations, which effectively misleads the signal prediction of NILM models to safeguard users' appliance-level privacy. The second challenge pertains to fundamental utility requirements, where existing adversarial attack schemes struggle to achieve accurate billing calculation for users. To handle this problem, we introduce an additional constraint, mandating that the sum of added perturbations within a billing period must be precisely zero. Experimental validation on real-world power datasets REDD and UK-DALE demonstrates the efficacy of our proposed solutions, which can significantly amplify the discrepancy between the output of the targeted NILM model and the actual power signal of appliances, and enable accurate billing at the same time. Additionally, our solutions exhibit transferability, making the generated perturbation signal from one target model applicable to other diverse NILM models.



## **6. Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks**

cs.IR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16708v1) [paper-pdf](http://arxiv.org/pdf/2412.16708v1)

**Authors**: Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, Claire Cardie

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a promising solution to mitigate LLM hallucinations and enhance their performance in knowledge-intensive domains. However, these systems are vulnerable to adversarial poisoning attacks, where malicious passages injected into retrieval databases can mislead the model into generating factually incorrect outputs. In this paper, we investigate both the retrieval and the generation components of RAG systems to understand how to enhance their robustness against such attacks. From the retrieval perspective, we analyze why and how the adversarial contexts are retrieved and assess how the quality of the retrieved passages impacts downstream generation. From a generation perspective, we evaluate whether LLMs' advanced critical thinking and internal knowledge capabilities can be leveraged to mitigate the impact of adversarial contexts, i.e., using skeptical prompting as a self-defense mechanism. Our experiments and findings provide actionable insights into designing safer and more resilient retrieval-augmented frameworks, paving the way for their reliable deployment in real-world applications.



## **7. Adversarial Attack Against Images Classification based on Generative Adversarial Networks**

cs.CV

7 pages, 6 figures

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16662v1) [paper-pdf](http://arxiv.org/pdf/2412.16662v1)

**Authors**: Yahe Yang

**Abstract**: Adversarial attacks on image classification systems have always been an important problem in the field of machine learning, and generative adversarial networks (GANs), as popular models in the field of image generation, have been widely used in various novel scenarios due to their powerful generative capabilities. However, with the popularity of generative adversarial networks, the misuse of fake image technology has raised a series of security problems, such as malicious tampering with other people's photos and videos, and invasion of personal privacy. Inspired by the generative adversarial networks, this work proposes a novel adversarial attack method, aiming to gain insight into the weaknesses of the image classification system and improve its anti-attack ability. Specifically, the generative adversarial networks are used to generate adversarial samples with small perturbations but enough to affect the decision-making of the classifier, and the adversarial samples are generated through the adversarial learning of the training generator and the classifier. From extensive experiment analysis, we evaluate the effectiveness of the method on a classical image classification dataset, and the results show that our model successfully deceives a variety of advanced classifiers while maintaining the naturalness of adversarial samples.



## **8. PB-UAP: Hybrid Universal Adversarial Attack For Image Segmentation**

cs.CV

Accepted by ICASSP 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16651v1) [paper-pdf](http://arxiv.org/pdf/2412.16651v1)

**Authors**: Yufei Song, Ziqi Zhou, Minghui Li, Xianlong Wang, Menghao Deng, Wei Wan, Shengshan Hu, Leo Yu Zhang

**Abstract**: With the rapid advancement of deep learning, the model robustness has become a significant research hotspot, \ie, adversarial attacks on deep neural networks. Existing works primarily focus on image classification tasks, aiming to alter the model's predicted labels. Due to the output complexity and deeper network architectures, research on adversarial examples for segmentation models is still limited, particularly for universal adversarial perturbations. In this paper, we propose a novel universal adversarial attack method designed for segmentation models, which includes dual feature separation and low-frequency scattering modules. The two modules guide the training of adversarial examples in the pixel and frequency space, respectively. Experiments demonstrate that our method achieves high attack success rates surpassing the state-of-the-art methods, and exhibits strong transferability across different models.



## **9. POEX: Policy Executable Embodied AI Jailbreak Attacks**

cs.RO

Homepage: https://poex-eai-jailbreak.github.io/

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16633v1) [paper-pdf](http://arxiv.org/pdf/2412.16633v1)

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu

**Abstract**: The integration of large language models (LLMs) into the planning module of Embodied Artificial Intelligence (Embodied AI) systems has greatly enhanced their ability to translate complex user instructions into executable policies. In this paper, we demystified how traditional LLM jailbreak attacks behave in the Embodied AI context. We conducted a comprehensive safety analysis of the LLM-based planning module of embodied AI systems against jailbreak attacks. Using the carefully crafted Harmful-RLbench, we accessed 20 open-source and proprietary LLMs under traditional jailbreak attacks, and highlighted two key challenges when adopting the prior jailbreak techniques to embodied AI contexts: (1) The harmful text output by LLMs does not necessarily induce harmful policies in Embodied AI context, and (2) even we can generate harmful policies, we have to guarantee they are executable in practice. To overcome those challenges, we propose Policy Executable (POEX) jailbreak attacks, where harmful instructions and optimized suffixes are injected into LLM-based planning modules, leading embodied AI to perform harmful actions in both simulated and physical environments. Our approach involves constraining adversarial suffixes to evade detection and fine-tuning a policy evaluater to improve the executability of harmful policies. We conducted extensive experiments on both a robotic arm embodied AI platform and simulators, to validate the attack and policy success rates on 136 harmful instructions from Harmful-RLbench. Our findings expose serious safety vulnerabilities in LLM-based planning modules, including the ability of POEX to be transferred across models. Finally, we propose mitigation strategies, such as safety-constrained prompts, pre- and post-planning checks, to address these vulnerabilities and ensure the safe deployment of embodied AI in real-world settings.



## **10. Automated Progressive Red Teaming**

cs.CR

Accepted by COLING 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2407.03876v3) [paper-pdf](http://arxiv.org/pdf/2407.03876v3)

**Authors**: Bojian Jiang, Yi Jing, Tianhao Shen, Tong Wu, Qing Yang, Deyi Xiong

**Abstract**: Ensuring the safety of large language models (LLMs) is paramount, yet identifying potential vulnerabilities is challenging. While manual red teaming is effective, it is time-consuming, costly and lacks scalability. Automated red teaming (ART) offers a more cost-effective alternative, automatically generating adversarial prompts to expose LLM vulnerabilities. However, in current ART efforts, a robust framework is absent, which explicitly frames red teaming as an effectively learnable task. To address this gap, we propose Automated Progressive Red Teaming (APRT) as an effectively learnable framework. APRT leverages three core modules: an Intention Expanding LLM that generates diverse initial attack samples, an Intention Hiding LLM that crafts deceptive prompts, and an Evil Maker to manage prompt diversity and filter ineffective samples. The three modules collectively and progressively explore and exploit LLM vulnerabilities through multi-round interactions. In addition to the framework, we further propose a novel indicator, Attack Effectiveness Rate (AER) to mitigate the limitations of existing evaluation metrics. By measuring the likelihood of eliciting unsafe but seemingly helpful responses, AER aligns closely with human evaluations. Extensive experiments with both automatic and human evaluations, demonstrate the effectiveness of ARPT across both open- and closed-source LLMs. Specifically, APRT effectively elicits 54% unsafe yet useful responses from Meta's Llama-3-8B-Instruct, 50% from GPT-4o (API access), and 39% from Claude-3.5 (API access), showcasing its robust attack capability and transferability across LLMs (especially from open-source LLMs to closed-source LLMs).



## **11. PGD-Imp: Rethinking and Unleashing Potential of Classic PGD with Dual Strategies for Imperceptible Adversarial Attacks**

cs.LG

accepted by ICASSP 2025

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.11168v2) [paper-pdf](http://arxiv.org/pdf/2412.11168v2)

**Authors**: Jin Li, Zitong Yu, Ziqiang He, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks have recently attracted increasing research interests. Existing methods typically incorporate external modules or loss terms other than a simple $l_p$-norm into the attack process to achieve imperceptibility, while we argue that such additional designs may not be necessary. In this paper, we rethink the essence of imperceptible attacks and propose two simple yet effective strategies to unleash the potential of PGD, the common and classical attack, for imperceptibility from an optimization perspective. Specifically, the Dynamic Step Size is introduced to find the optimal solution with minimal attack cost towards the decision boundary of the attacked model, and the Adaptive Early Stop strategy is adopted to reduce the redundant strength of adversarial perturbations to the minimum level. The proposed PGD-Imperceptible (PGD-Imp) attack achieves state-of-the-art results in imperceptible adversarial attacks for both untargeted and targeted scenarios. When performing untargeted attacks against ResNet-50, PGD-Imp attains 100$\%$ (+0.3$\%$) ASR, 0.89 (-1.76) $l_2$ distance, and 52.93 (+9.2) PSNR with 57s (-371s) running time, significantly outperforming existing methods.



## **12. WiP: Deception-in-Depth Using Multiple Layers of Deception**

cs.CR

Presented at HoTSoS 2024

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2412.16430v1) [paper-pdf](http://arxiv.org/pdf/2412.16430v1)

**Authors**: Jason Landsborough, Neil C. Rowe, Thuy D. Nguyen, Sunny Fugate

**Abstract**: Deception is being increasingly explored as a cyberdefense strategy to protect operational systems. We are studying implementation of deception-in-depth strategies with initially three logical layers: network, host, and data. We draw ideas from military deception, network orchestration, software deception, file deception, fake honeypots, and moving-target defenses. We are building a prototype representing our ideas and will be testing it in several adversarial environments. We hope to show that deploying a broad range of deception techniques can be more effective in protecting systems than deploying single techniques. Unlike traditional deception methods that try to encourage active engagement from attackers to collect intelligence, we focus on deceptions that can be used on real machines to discourage attacks.



## **13. Chain-of-Scrutiny: Detecting Backdoor Attacks for Large Language Models**

cs.CR

**SubmitDate**: 2024-12-21    [abs](http://arxiv.org/abs/2406.05948v2) [paper-pdf](http://arxiv.org/pdf/2406.05948v2)

**Authors**: Xi Li, Yusen Zhang, Renze Lou, Chen Wu, Jiaqi Wang

**Abstract**: Large Language Models (LLMs), especially those accessed via APIs, have demonstrated impressive capabilities across various domains. However, users without technical expertise often turn to (untrustworthy) third-party services, such as prompt engineering, to enhance their LLM experience, creating vulnerabilities to adversarial threats like backdoor attacks. Backdoor-compromised LLMs generate malicious outputs to users when inputs contain specific "triggers" set by attackers. Traditional defense strategies, originally designed for small-scale models, are impractical for API-accessible LLMs due to limited model access, high computational costs, and data requirements. To address these limitations, we propose Chain-of-Scrutiny (CoS) which leverages LLMs' unique reasoning abilities to mitigate backdoor attacks. It guides the LLM to generate reasoning steps for a given input and scrutinizes for consistency with the final output -- any inconsistencies indicating a potential attack. It is well-suited for the popular API-only LLM deployments, enabling detection at minimal cost and with little data. User-friendly and driven by natural language, it allows non-experts to perform the defense independently while maintaining transparency. We validate the effectiveness of CoS through extensive experiments on various tasks and LLMs, with results showing greater benefits for more powerful LLMs.



## **14. EMPRA: Embedding Perturbation Rank Attack against Neural Ranking Models**

cs.IR

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.16382v1) [paper-pdf](http://arxiv.org/pdf/2412.16382v1)

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke

**Abstract**: Recent research has shown that neural information retrieval techniques may be susceptible to adversarial attacks. Adversarial attacks seek to manipulate the ranking of documents, with the intention of exposing users to targeted content. In this paper, we introduce the Embedding Perturbation Rank Attack (EMPRA) method, a novel approach designed to perform adversarial attacks on black-box Neural Ranking Models (NRMs). EMPRA manipulates sentence-level embeddings, guiding them towards pertinent context related to the query while preserving semantic integrity. This process generates adversarial texts that seamlessly integrate with the original content and remain imperceptible to humans. Our extensive evaluation conducted on the widely-used MS MARCO V1 passage collection demonstrate the effectiveness of EMPRA against a wide range of state-of-the-art baselines in promoting a specific set of target documents within a given ranked results. Specifically, EMPRA successfully achieves a re-ranking of almost 96% of target documents originally ranked between 51-100 to rank within the top 10. Furthermore, EMPRA does not depend on surrogate models for adversarial text generation, enhancing its robustness against different NRMs in realistic settings.



## **15. Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context**

cs.CL

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.16359v1) [paper-pdf](http://arxiv.org/pdf/2412.16359v1)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on LLM vulnerabilities often relied on nonsensical adversarial prompts, which were easily detectable by automated methods. We address this gap by focusing on human-readable adversarial prompts, a more realistic and potent threat. Our key contributions are situation-driven attacks leveraging movie scripts to create contextually relevant, human-readable prompts that successfully deceive LLMs, adversarial suffix conversion to transform nonsensical adversarial suffixes into meaningful text, and AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B. Our findings demonstrate that LLMs can be tricked by sophisticated adversaries into producing harmful responses with human-readable adversarial prompts and that there exists a scope for improvement when it comes to robust LLMs.



## **16. Texture- and Shape-based Adversarial Attacks for Vehicle Detection in Synthetic Overhead Imagery**

cs.CV

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.16358v1) [paper-pdf](http://arxiv.org/pdf/2412.16358v1)

**Authors**: Mikael Yeghiazaryan, Sai Abhishek Siddhartha Namburu, Emily Kim, Stanislav Panev, Celso de Melo, Brent Lance, Fernando De la Torre, Jessica K. Hodgins

**Abstract**: Detecting vehicles in aerial images can be very challenging due to complex backgrounds, small resolution, shadows, and occlusions. Despite the effectiveness of SOTA detectors such as YOLO, they remain vulnerable to adversarial attacks (AAs), compromising their reliability. Traditional AA strategies often overlook the practical constraints of physical implementation, focusing solely on attack performance. Our work addresses this issue by proposing practical implementation constraints for AA in texture and/or shape. These constraints include pixelation, masking, limiting the color palette of the textures, and constraining the shape modifications. We evaluated the proposed constraints through extensive experiments using three widely used object detector architectures, and compared them to previous works. The results demonstrate the effectiveness of our solutions and reveal a trade-off between practicality and performance. Additionally, we introduce a labeled dataset of overhead images featuring vehicles of various categories. We will make the code/dataset public upon paper acceptance.



## **17. Competence-Based Analysis of Language Models**

cs.CL

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2303.00333v5) [paper-pdf](http://arxiv.org/pdf/2303.00333v5)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain behaviors across these tasks.



## **18. Augment then Smooth: Reconciling Differential Privacy with Certified Robustness**

cs.LG

29 pages, 19 figures. Accepted at TMLR in 2024. Link:  https://openreview.net/forum?id=YN0IcnXqsr

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2306.08656v3) [paper-pdf](http://arxiv.org/pdf/2306.08656v3)

**Authors**: Jiapeng Wu, Atiyeh Ashari Ghomi, David Glukhov, Jesse C. Cresswell, Franziska Boenisch, Nicolas Papernot

**Abstract**: Machine learning models are susceptible to a variety of attacks that can erode trust, including attacks against the privacy of training data, and adversarial examples that jeopardize model accuracy. Differential privacy and certified robustness are effective frameworks for combating these two threats respectively, as they each provide future-proof guarantees. However, we show that standard differentially private model training is insufficient for providing strong certified robustness guarantees. Indeed, combining differential privacy and certified robustness in a single system is non-trivial, leading previous works to introduce complex training schemes that lack flexibility. In this work, we present DP-CERT, a simple and effective method that achieves both privacy and robustness guarantees simultaneously by integrating randomized smoothing into standard differentially private model training. Compared to the leading prior work, DP-CERT gives up to a 2.5% increase in certified accuracy for the same differential privacy guarantee on CIFAR10. Through in-depth per-sample metric analysis, we find that larger certifiable radii correlate with smaller local Lipschitz constants, and show that DP-CERT effectively reduces Lipschitz constants compared to other differentially private training methods. The code is available at github.com/layer6ai-labs/dp-cert.



## **19. Watertox: The Art of Simplicity in Universal Attacks A Cross-Model Framework for Robust Adversarial Generation**

cs.CV

18 pages, 4 figures, 3 tables. Advances a novel method for generating  cross-model transferable adversarial perturbations through a two-stage FGSM  process and architectural ensemble voting mechanism

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.15924v1) [paper-pdf](http://arxiv.org/pdf/2412.15924v1)

**Authors**: Zhenghao Gao, Shengjie Xu, Meixi Chen, Fangyao Zhao

**Abstract**: Contemporary adversarial attack methods face significant limitations in cross-model transferability and practical applicability. We present Watertox, an elegant adversarial attack framework achieving remarkable effectiveness through architectural diversity and precision-controlled perturbations. Our two-stage Fast Gradient Sign Method combines uniform baseline perturbations ($\epsilon_1 = 0.1$) with targeted enhancements ($\epsilon_2 = 0.4$). The framework leverages an ensemble of complementary architectures, from VGG to ConvNeXt, synthesizing diverse perspectives through an innovative voting mechanism. Against state-of-the-art architectures, Watertox reduces model accuracy from 70.6% to 16.0%, with zero-shot attacks achieving up to 98.8% accuracy reduction against unseen architectures. These results establish Watertox as a significant advancement in adversarial methodologies, with promising applications in visual security systems and CAPTCHA generation.



## **20. Client-Side Patching against Backdoor Attacks in Federated Learning**

cs.CR

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.10605v2) [paper-pdf](http://arxiv.org/pdf/2412.10605v2)

**Authors**: Borja Molina-Coronado

**Abstract**: Federated learning is a versatile framework for training models in decentralized environments. However, the trust placed in clients makes federated learning vulnerable to backdoor attacks launched by malicious participants. While many defenses have been proposed, they often fail short when facing heterogeneous data distributions among participating clients. In this paper, we propose a novel defense mechanism for federated learning systems designed to mitigate backdoor attacks on the clients-side. Our approach leverages adversarial learning techniques and model patching to neutralize the impact of backdoor attacks. Through extensive experiments on the MNIST and Fashion-MNIST datasets, we demonstrate that our defense effectively reduces backdoor accuracy, outperforming existing state-of-the-art defenses, such as LFighter, FLAME, and RoseAgg, in i.i.d. and non-i.i.d. scenarios, while maintaining competitive or superior accuracy on clean data.



## **21. PoisonCatcher: Revealing and Identifying LDP Poisoning Attacks in IIoT**

cs.CR

12 pages,5 figures, 3 tables

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.15704v1) [paper-pdf](http://arxiv.org/pdf/2412.15704v1)

**Authors**: Lisha Shuai, Shaofeng Tan, Nan Zhang, Jiamin Zhang, Min Zhang, Xiaolong Yang

**Abstract**: Local Differential Privacy (LDP) is widely adopted in the Industrial Internet of Things (IIoT) for its lightweight, decentralized, and scalable nature. However, its perturbation-based privacy mechanism makes it difficult to distinguish between uncontaminated and tainted data, encouraging adversaries to launch poisoning attacks. While LDP provides some resilience against minor poisoning, it lacks robustness in IIoT with dynamic networks and substantial real-time data flows. Effective countermeasures for such attacks are still underdeveloped. This work narrows the critical gap by revealing and identifying LDP poisoning attacks in IIoT. We begin by deepening the understanding of such attacks, revealing novel threats that arise from the interplay between LDP indistinguishability and IIoT complexity. This exploration uncovers a novel rule-poisoning attack, and presents a general attack formulation by unifying it with input-poisoning and output-poisoning. Furthermore, two key attack impacts, i.e., Statistical Query Result (SQR) accuracy degradation and inter-dataset correlations disruption, along with two characteristics: attack patterns unstable and poisoned data stealth are revealed. From this, we propose PoisonCatcher, a four-stage solution that detects LDP poisoning attacks and identifies specific contaminated data points. It utilizes temporal similarity, attribute correlation, and time-series stability analysis to detect datasets exhibiting SQR accuracy degradation, inter-dataset disruptions, and unstable patterns. Enhanced feature engineering is used to extract subtle poisoning signatures, enabling machine learning models to identify specific contamination. Experimental evaluations show the effectiveness, achieving state-of-the-art performance with average precision and recall rates of 86.17% and 97.5%, respectively, across six representative attack scenarios.



## **22. CAMH: Advancing Model Hijacking Attack in Machine Learning**

cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2408.13741v2) [paper-pdf](http://arxiv.org/pdf/2408.13741v2)

**Authors**: Xing He, Jiahao Chen, Yuwen Pu, Qingming Li, Chunyi Zhou, Yingcai Wu, Jinbao Li, Shouling Ji

**Abstract**: In the burgeoning domain of machine learning, the reliance on third-party services for model training and the adoption of pre-trained models have surged. However, this reliance introduces vulnerabilities to model hijacking attacks, where adversaries manipulate models to perform unintended tasks, leading to significant security and ethical concerns, like turning an ordinary image classifier into a tool for detecting faces in pornographic content, all without the model owner's knowledge. This paper introduces Category-Agnostic Model Hijacking (CAMH), a novel model hijacking attack method capable of addressing the challenges of class number mismatch, data distribution divergence, and performance balance between the original and hijacking tasks. CAMH incorporates synchronized training layers, random noise optimization, and a dual-loop optimization approach to ensure minimal impact on the original task's performance while effectively executing the hijacking task. We evaluate CAMH across multiple benchmark datasets and network architectures, demonstrating its potent attack effectiveness while ensuring minimal degradation in the performance of the original task.



## **23. A Stealthy Wrongdoer: Feature-Oriented Reconstruction Attack against Split Learning**

cs.CR

Accepted to CVPR 2024

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2405.04115v3) [paper-pdf](http://arxiv.org/pdf/2405.04115v3)

**Authors**: Xiaoyang Xu, Mengda Yang, Wenzhe Yi, Ziang Li, Juan Wang, Hongxin Hu, Yong Zhuang, Yaxin Liu

**Abstract**: Split Learning (SL) is a distributed learning framework renowned for its privacy-preserving features and minimal computational requirements. Previous research consistently highlights the potential privacy breaches in SL systems by server adversaries reconstructing training data. However, these studies often rely on strong assumptions or compromise system utility to enhance attack performance. This paper introduces a new semi-honest Data Reconstruction Attack on SL, named Feature-Oriented Reconstruction Attack (FORA). In contrast to prior works, FORA relies on limited prior knowledge, specifically that the server utilizes auxiliary samples from the public without knowing any client's private information. This allows FORA to conduct the attack stealthily and achieve robust performance. The key vulnerability exploited by FORA is the revelation of the model representation preference in the smashed data output by victim client. FORA constructs a substitute client through feature-level transfer learning, aiming to closely mimic the victim client's representation preference. Leveraging this substitute client, the server trains the attack model to effectively reconstruct private data. Extensive experiments showcase FORA's superior performance compared to state-of-the-art methods. Furthermore, the paper systematically evaluates the proposed method's applicability across diverse settings and advanced defense strategies.



## **24. JailPO: A Novel Black-box Jailbreak Framework via Preference Optimization against Aligned LLMs**

cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.15623v1) [paper-pdf](http://arxiv.org/pdf/2412.15623v1)

**Authors**: Hongyi Li, Jiawei Ye, Jie Wu, Tianjie Yan, Chu Wang, Zhixin Li

**Abstract**: Large Language Models (LLMs) aligned with human feedback have recently garnered significant attention. However, it remains vulnerable to jailbreak attacks, where adversaries manipulate prompts to induce harmful outputs. Exploring jailbreak attacks enables us to investigate the vulnerabilities of LLMs and further guides us in enhancing their security. Unfortunately, existing techniques mainly rely on handcrafted templates or generated-based optimization, posing challenges in scalability, efficiency and universality. To address these issues, we present JailPO, a novel black-box jailbreak framework to examine LLM alignment. For scalability and universality, JailPO meticulously trains attack models to automatically generate covert jailbreak prompts. Furthermore, we introduce a preference optimization-based attack method to enhance the jailbreak effectiveness, thereby improving efficiency. To analyze model vulnerabilities, we provide three flexible jailbreak patterns. Extensive experiments demonstrate that JailPO not only automates the attack process while maintaining effectiveness but also exhibits superior performance in efficiency, universality, and robustness against defenses compared to baselines. Additionally, our analysis of the three JailPO patterns reveals that attacks based on complex templates exhibit higher attack strength, whereas covert question transformations elicit riskier responses and are more likely to bypass defense mechanisms.



## **25. Adversarial Robustness through Dynamic Ensemble Learning**

cs.CR

This is the accepted version of our paper for the 2024 IEEE Silchar  Subsection Conference (IEEE SILCON24), held from November 15 to 17, 2024, at  the National Institute of Technology (NIT), Agartala, India. The paper is 6  pages long and contains 3 Figures and 7 Tables

**SubmitDate**: 2024-12-20    [abs](http://arxiv.org/abs/2412.16254v1) [paper-pdf](http://arxiv.org/pdf/2412.16254v1)

**Authors**: Hetvi Waghela, Jaydip Sen, Sneha Rakshit

**Abstract**: Adversarial attacks pose a significant threat to the reliability of pre-trained language models (PLMs) such as GPT, BERT, RoBERTa, and T5. This paper presents Adversarial Robustness through Dynamic Ensemble Learning (ARDEL), a novel scheme designed to enhance the robustness of PLMs against such attacks. ARDEL leverages the diversity of multiple PLMs and dynamically adjusts the ensemble configuration based on input characteristics and detected adversarial patterns. Key components of ARDEL include a meta-model for dynamic weighting, an adversarial pattern detection module, and adversarial training with regularization techniques. Comprehensive evaluations using standardized datasets and various adversarial attack scenarios demonstrate that ARDEL significantly improves robustness compared to existing methods. By dynamically reconfiguring the ensemble to prioritize the most robust models for each input, ARDEL effectively reduces attack success rates and maintains higher accuracy under adversarial conditions. This work contributes to the broader goal of developing more secure and trustworthy AI systems for real-world NLP applications, offering a practical and scalable solution to enhance adversarial resilience in PLMs.



## **26. Time Will Tell: Timing Side Channels via Output Token Count in Large Language Models**

cs.LG

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.15431v1) [paper-pdf](http://arxiv.org/pdf/2412.15431v1)

**Authors**: Tianchen Zhang, Gururaj Saileshwar, David Lie

**Abstract**: This paper demonstrates a new side-channel that enables an adversary to extract sensitive information about inference inputs in large language models (LLMs) based on the number of output tokens in the LLM response. We construct attacks using this side-channel in two common LLM tasks: recovering the target language in machine translation tasks and recovering the output class in classification tasks. In addition, due to the auto-regressive generation mechanism in LLMs, an adversary can recover the output token count reliably using a timing channel, even over the network against a popular closed-source commercial LLM. Our experiments show that an adversary can learn the output language in translation tasks with more than 75% precision across three different models (Tower, M2M100, MBart50). Using this side-channel, we also show the input class in text classification tasks can be leaked out with more than 70% precision from open-source LLMs like Llama-3.1, Llama-3.2, Gemma2, and production models like GPT-4o. Finally, we propose tokenizer-, system-, and prompt-based mitigations against the output token count side-channel.



## **27. Towards Adversarially Robust Dataset Distillation by Curvature Regularization**

cs.LG

17 pages, 3 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2403.10045v2) [paper-pdf](http://arxiv.org/pdf/2403.10045v2)

**Authors**: Eric Xue, Yijiang Li, Haoyang Liu, Peiran Wang, Yifan Shen, Haohan Wang

**Abstract**: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accuracy and robustness with less computation overhead but is also capable of generating robust distilled datasets that can withstand various adversarial attacks.



## **28. AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving**

cs.CV

55 pages, 14 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.15206v1) [paper-pdf](http://arxiv.org/pdf/2412.15206v1)

**Authors**: Shuo Xing, Hongyuan Hua, Xiangbo Gao, Shenzhe Zhu, Renjie Li, Kexin Tian, Xiaopeng Li, Heng Huang, Tianbao Yang, Zhangyang Wang, Yang Zhou, Huaxiu Yao, Zhengzhong Tu

**Abstract**: Recent advancements in large vision language models (VLMs) tailored for autonomous driving (AD) have shown strong scene understanding and reasoning capabilities, making them undeniable candidates for end-to-end driving systems. However, limited work exists on studying the trustworthiness of DriveVLMs -- a critical factor that directly impacts public transportation safety. In this paper, we introduce AutoTrust, a comprehensive trustworthiness benchmark for large vision-language models in autonomous driving (DriveVLMs), considering diverse perspectives -- including trustfulness, safety, robustness, privacy, and fairness. We constructed the largest visual question-answering dataset for investigating trustworthiness issues in driving scenarios, comprising over 10k unique scenes and 18k queries. We evaluated six publicly available VLMs, spanning from generalist to specialist, from open-source to commercial models. Our exhaustive evaluations have unveiled previously undiscovered vulnerabilities of DriveVLMs to trustworthiness threats. Specifically, we found that the general VLMs like LLaVA-v1.6 and GPT-4o-mini surprisingly outperform specialized models fine-tuned for driving in terms of overall trustworthiness. DriveVLMs like DriveLM-Agent are particularly vulnerable to disclosing sensitive information. Additionally, both generalist and specialist VLMs remain susceptible to adversarial attacks and struggle to ensure unbiased decision-making across diverse environments and populations. Our findings call for immediate and decisive action to address the trustworthiness of DriveVLMs -- an issue of critical importance to public safety and the welfare of all citizens relying on autonomous transportation systems. Our benchmark is publicly available at \url{https://github.com/taco-group/AutoTrust}, and the leaderboard is released at \url{https://taco-group.github.io/AutoTrust/}.



## **29. Do Parameters Reveal More than Loss for Membership Inference?**

cs.LG

Accepted to Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2406.11544v4) [paper-pdf](http://arxiv.org/pdf/2406.11544v4)

**Authors**: Anshuman Suri, Xiao Zhang, David Evans

**Abstract**: Membership inference attacks are used as a key tool for disclosure auditing. They aim to infer whether an individual record was used to train a model. While such evaluations are useful to demonstrate risk, they are computationally expensive and often make strong assumptions about potential adversaries' access to models and training environments, and thus do not provide tight bounds on leakage from potential attacks. We show how prior claims around black-box access being sufficient for optimal membership inference do not hold for stochastic gradient descent, and that optimal membership inference indeed requires white-box access. Our theoretical results lead to a new white-box inference attack, IHA (Inverse Hessian Attack), that explicitly uses model parameters by taking advantage of computing inverse-Hessian vector products. Our results show that both auditors and adversaries may be able to benefit from access to model parameters, and we advocate for further research into white-box methods for membership inference.



## **30. Accuracy Limits as a Barrier to Biometric System Security**

cs.CR

14 pages, 4 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.13099v2) [paper-pdf](http://arxiv.org/pdf/2412.13099v2)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Pascal Lafourcade, Kevin Thiry-Atighehchi

**Abstract**: Biometric systems are widely used for identity verification and identification, including authentication (i.e., one-to-one matching to verify a claimed identity) and identification (i.e., one-to-many matching to find a subject in a database). The matching process relies on measuring similarities or dissimilarities between a fresh biometric template and enrolled templates. The False Match Rate FMR is a key metric for assessing the accuracy and reliability of such systems. This paper analyzes biometric systems based on their FMR, with two main contributions. First, we explore untargeted attacks, where an adversary aims to impersonate any user within a database. We determine the number of trials required for an attacker to successfully impersonate a user and derive the critical population size (i.e., the maximum number of users in the database) required to maintain a given level of security. Furthermore, we compute the critical FMR value needed to ensure resistance against untargeted attacks as the database size increases. Second, we revisit the biometric birthday problem to evaluate the approximate and exact probabilities that two users in a database collide (i.e., can impersonate each other). Based on this analysis, we derive both the approximate critical population size and the critical FMR value needed to bound the likelihood of such collisions occurring with a given probability. These thresholds offer insights for designing systems that mitigate the risk of impersonation and collisions, particularly in large-scale biometric databases. Our findings indicate that current biometric systems fail to deliver sufficient accuracy to achieve an adequate security level against untargeted attacks, even in small-scale databases. Moreover, state-of-the-art systems face significant challenges in addressing the biometric birthday problem, especially as database sizes grow.



## **31. SLIFER: Investigating Performance and Robustness of Malware Detection Pipelines**

cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2405.14478v3) [paper-pdf](http://arxiv.org/pdf/2405.14478v3)

**Authors**: Andrea Ponte, Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Ivan Tesfai Ogbu, Fabio Roli

**Abstract**: As a result of decades of research, Windows malware detection is approached through a plethora of techniques. However, there is an ongoing mismatch between academia -- which pursues an optimal performances in terms of detection rate and low false alarms -- and the requirements of real-world scenarios. In particular, academia focuses on combining static and dynamic analysis within a single or ensemble of models, falling into several pitfalls like (i) firing dynamic analysis without considering the computational burden it requires; (ii) discarding impossible-to-analyze samples; and (iii) analyzing robustness against adversarial attacks without considering that malware detectors are complemented with more non-machine-learning components. Thus, in this paper we bridge these gaps, by investigating the properties of malware detectors built with multiple and different types of analysis. To do so, we develop SLIFER, a Windows malware detection pipeline sequentially leveraging both static and dynamic analysis, interrupting computations as soon as one module triggers an alarm, requiring dynamic analysis only when needed. Contrary to the state of the art, we investigate how to deal with samples that impede analyzes, showing how much they impact performances, concluding that it is better to flag them as legitimate to not drastically increase false alarms. Lastly, we perform a robustness evaluation of SLIFER. Counter-intuitively, the injection of new content is either blocked more by signatures than dynamic analysis, due to byte artifacts created by the attack, or it is able to avoid detection from signatures, as they rely on constraints on file size disrupted by attacks. As far as we know, we are the first to investigate the properties of sequential malware detectors, shedding light on their behavior in real production environment.



## **32. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2410.23091v5) [paper-pdf](http://arxiv.org/pdf/2410.23091v5)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff



## **33. Grimm: A Plug-and-Play Perturbation Rectifier for Graph Neural Networks Defending against Poisoning Attacks**

cs.LG

19 pages, 13 figures

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08555v2) [paper-pdf](http://arxiv.org/pdf/2412.08555v2)

**Authors**: Ao Liu, Wenshan Li, Beibei Li, Wengang Ma, Tao Li, Pan Zhou

**Abstract**: Recent studies have revealed the vulnerability of graph neural networks (GNNs) to adversarial poisoning attacks on node classification tasks. Current defensive methods require substituting the original GNNs with defense models, regardless of the original's type. This approach, while targeting adversarial robustness, compromises the enhancements developed in prior research to boost GNNs' practical performance. Here we introduce Grimm, the first plug-and-play defense model. With just a minimal interface requirement for extracting features from any layer of the protected GNNs, Grimm is thus enabled to seamlessly rectify perturbations. Specifically, we utilize the feature trajectories (FTs) generated by GNNs, as they evolve through epochs, to reflect the training status of the networks. We then theoretically prove that the FTs of victim nodes will inevitably exhibit discriminable anomalies. Consequently, inspired by the natural parallelism between the biological nervous and immune systems, we construct Grimm, a comprehensive artificial immune system for GNNs. Grimm not only detects abnormal FTs and rectifies adversarial edges during training but also operates efficiently in parallel, thereby mirroring the concurrent functionalities of its biological counterparts. We experimentally confirm that Grimm offers four empirically validated advantages: 1) Harmlessness, as it does not actively interfere with GNN training; 2) Parallelism, ensuring monitoring, detection, and rectification functions operate independently of the GNN training process; 3) Generalizability, demonstrating compatibility with mainstream GNNs such as GCN, GAT, and GraphSAGE; and 4) Transferability, as the detectors for abnormal FTs can be efficiently transferred across different systems for one-step rectification.



## **34. DG-Mamba: Robust and Efficient Dynamic Graph Structure Learning with Selective State Space Models**

cs.LG

Accepted by the Main Technical Track of the 39th Annual AAAI  Conference on Artificial Intelligence (AAAI-2025)

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08160v4) [paper-pdf](http://arxiv.org/pdf/2412.08160v4)

**Authors**: Haonan Yuan, Qingyun Sun, Zhaonan Wang, Xingcheng Fu, Cheng Ji, Yongjian Wang, Bo Jin, Jianxin Li

**Abstract**: Dynamic graphs exhibit intertwined spatio-temporal evolutionary patterns, widely existing in the real world. Nevertheless, the structure incompleteness, noise, and redundancy result in poor robustness for Dynamic Graph Neural Networks (DGNNs). Dynamic Graph Structure Learning (DGSL) offers a promising way to optimize graph structures. However, aside from encountering unacceptable quadratic complexity, it overly relies on heuristic priors, making it hard to discover underlying predictive patterns. How to efficiently refine the dynamic structures, capture intrinsic dependencies, and learn robust representations, remains under-explored. In this work, we propose the novel DG-Mamba, a robust and efficient Dynamic Graph structure learning framework with the Selective State Space Models (Mamba). To accelerate the spatio-temporal structure learning, we propose a kernelized dynamic message-passing operator that reduces the quadratic time complexity to linear. To capture global intrinsic dynamics, we establish the dynamic graph as a self-contained system with State Space Model. By discretizing the system states with the cross-snapshot graph adjacency, we enable the long-distance dependencies capturing with the selective snapshot scan. To endow learned dynamic structures more expressive with informativeness, we propose the self-supervised Principle of Relevant Information for DGSL to regularize the most relevant yet least redundant information, enhancing global robustness. Extensive experiments demonstrate the superiority of the robustness and efficiency of our DG-Mamba compared with the state-of-the-art baselines against adversarial attacks.



## **35. How Does the Smoothness Approximation Method Facilitate Generalization for Federated Adversarial Learning?**

cs.LG

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08282v2) [paper-pdf](http://arxiv.org/pdf/2412.08282v2)

**Authors**: Wenjun Ding, Ying An, Lixing Chen, Shichao Kan, Fan Wu, Zhe Qu

**Abstract**: Federated Adversarial Learning (FAL) is a robust framework for resisting adversarial attacks on federated learning. Although some FAL studies have developed efficient algorithms, they primarily focus on convergence performance and overlook generalization. Generalization is crucial for evaluating algorithm performance on unseen data. However, generalization analysis is more challenging due to non-smooth adversarial loss functions. A common approach to addressing this issue is to leverage smoothness approximation. In this paper, we develop algorithm stability measures to evaluate the generalization performance of two popular FAL algorithms: \textit{Vanilla FAL (VFAL)} and {\it Slack FAL (SFAL)}, using three different smooth approximation methods: 1) \textit{Surrogate Smoothness Approximation (SSA)}, (2) \textit{Randomized Smoothness Approximation (RSA)}, and (3) \textit{Over-Parameterized Smoothness Approximation (OPSA)}. Based on our in-depth analysis, we answer the question of how to properly set the smoothness approximation method to mitigate generalization error in FAL. Moreover, we identify RSA as the most effective method for reducing generalization error. In highly data-heterogeneous scenarios, we also recommend employing SFAL to mitigate the deterioration of generalization performance caused by heterogeneity. Based on our theoretical results, we provide insights to help develop more efficient FAL algorithms, such as designing new metrics and dynamic aggregation rules to mitigate heterogeneity.



## **36. Unleashing the Unseen: Harnessing Benign Datasets for Jailbreaking Large Language Models**

cs.CR

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2410.00451v3) [paper-pdf](http://arxiv.org/pdf/2410.00451v3)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Despite significant ongoing efforts in safety alignment, large language models (LLMs) such as GPT-4 and LLaMA 3 remain vulnerable to jailbreak attacks that can induce harmful behaviors, including through the use of adversarial suffixes. Building on prior research, we hypothesize that these adversarial suffixes are not mere bugs but may represent features that can dominate the LLM's behavior. To evaluate this hypothesis, we conduct several experiments. First, we demonstrate that benign features can be effectively made to function as adversarial suffixes, i.e., we develop a feature extraction method to extract sample-agnostic features from benign dataset in the form of suffixes and show that these suffixes may effectively compromise safety alignment. Second, we show that adversarial suffixes generated from jailbreak attacks may contain meaningful features, i.e., appending the same suffix to different prompts results in responses exhibiting specific characteristics. Third, we show that such benign-yet-safety-compromising features can be easily introduced through fine-tuning using only benign datasets. As a result, we are able to completely eliminate GPT's safety alignment in a blackbox setting through finetuning with only benign data. Our code and data is available at \url{https://github.com/suffix-maybe-feature/adver-suffix-maybe-features}.



## **37. Doubly-Universal Adversarial Perturbations: Deceiving Vision-Language Models Across Both Images and Text with a Single Perturbation**

cs.CV

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.08108v2) [paper-pdf](http://arxiv.org/pdf/2412.08108v2)

**Authors**: Hee-Seon Kim, Minbeom Kim, Changick Kim

**Abstract**: Large Vision-Language Models (VLMs) have demonstrated remarkable performance across multimodal tasks by integrating vision encoders with large language models (LLMs). However, these models remain vulnerable to adversarial attacks. Among such attacks, Universal Adversarial Perturbations (UAPs) are especially powerful, as a single optimized perturbation can mislead the model across various input images. In this work, we introduce a novel UAP specifically designed for VLMs: the Doubly-Universal Adversarial Perturbation (Doubly-UAP), capable of universally deceiving VLMs across both image and text inputs. To successfully disrupt the vision encoder's fundamental process, we analyze the core components of the attention mechanism. After identifying value vectors in the middle-to-late layers as the most vulnerable, we optimize Doubly-UAP in a label-free manner with a frozen model. Despite being developed as a black-box to the LLM, Doubly-UAP achieves high attack success rates on VLMs, consistently outperforming baseline methods across vision-language tasks. Extensive ablation studies and analyses further demonstrate the robustness of Doubly-UAP and provide insights into how it influences internal attention mechanisms.



## **38. Towards Provable Security in Industrial Control Systems Via Dynamic Protocol Attestation**

cs.CR

This paper was accepted into the ICSS'24 workshop

**SubmitDate**: 2024-12-19    [abs](http://arxiv.org/abs/2412.14467v1) [paper-pdf](http://arxiv.org/pdf/2412.14467v1)

**Authors**: Arthur Amorim, Trevor Kann, Max Taylor, Lance Joneckis

**Abstract**: Industrial control systems (ICSs) increasingly rely on digital technologies vulnerable to cyber attacks. Cyber attackers can infiltrate ICSs and execute malicious actions. Individually, each action seems innocuous. But taken together, they cause the system to enter an unsafe state. These attacks have resulted in dramatic consequences such as physical damage, economic loss, and environmental catastrophes. This paper introduces a methodology that restricts actions using protocols. These protocols only allow safe actions to execute. Protocols are written in a domain specific language we have embedded in an interactive theorem prover (ITP). The ITP enables formal, machine-checked proofs to ensure protocols maintain safety properties. We use dynamic attestation to ensure ICSs conform to their protocol even if an adversary compromises a component. Since protocol conformance prevents unsafe actions, the previously mentioned cyber attacks become impossible. We demonstrate the effectiveness of our methodology using an example from the Fischertechnik Industry 4.0 platform. We measure dynamic attestation's impact on latency and throughput. Our approach is a starting point for studying how to combine formal methods and protocol design to thwart attacks intended to cripple ICSs.



## **39. Adversarial Hubness in Multi-Modal Retrieval**

cs.CR

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.14113v1) [paper-pdf](http://arxiv.org/pdf/2412.14113v1)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries. In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts. We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system based on a tutorial from Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries). We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.



## **40. Certification of Speaker Recognition Models to Additive Perturbations**

cs.SD

13 pages, 10 figures; AAAI-2025 accepted paper

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2404.18791v2) [paper-pdf](http://arxiv.org/pdf/2404.18791v2)

**Authors**: Dmitrii Korzh, Elvir Karimov, Mikhail Pautov, Oleg Y. Rogov, Ivan Oseledets

**Abstract**: Speaker recognition technology is applied to various tasks, from personal virtual assistants to secure access systems. However, the robustness of these systems against adversarial attacks, particularly to additive perturbations, remains a significant challenge. In this paper, we pioneer applying robustness certification techniques to speaker recognition, initially developed for the image domain. Our work covers this gap by transferring and improving randomized smoothing certification techniques against norm-bounded additive perturbations for classification and few-shot learning tasks to speaker recognition. We demonstrate the effectiveness of these methods on VoxCeleb 1 and 2 datasets for several models. We expect this work to improve the robustness of voice biometrics and accelerate the research of certification methods in the audio domain.



## **41. Adversarial Robustness of Link Sign Prediction in Signed Graphs**

cs.LG

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2401.10590v2) [paper-pdf](http://arxiv.org/pdf/2401.10590v2)

**Authors**: Jialong Zhou, Xing Ai, Yuni Lai, Tomasz Michalak, Gaolei Li, Jianhua Li, Kai Zhou

**Abstract**: Signed graphs serve as fundamental data structures for representing positive and negative relationships in social networks, with signed graph neural networks (SGNNs) emerging as the primary tool for their analysis. Our investigation reveals that balance theory, while essential for modeling signed relationships in SGNNs, inadvertently introduces exploitable vulnerabilities to black-box attacks. To demonstrate this vulnerability, we propose balance-attack, a novel adversarial strategy specifically designed to compromise graph balance degree, and develop an efficient heuristic algorithm to solve the associated NP-hard optimization problem. While existing approaches attempt to restore attacked graphs through balance learning techniques, they face a critical challenge we term "Irreversibility of Balance-related Information," where restored edges fail to align with original attack targets. To address this limitation, we introduce Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), an innovative framework that combines contrastive learning with balance augmentation techniques to achieve robust graph representations. By maintaining high balance degree in the latent space, BA-SGCL effectively circumvents the irreversibility challenge and enhances model resilience. Extensive experiments across multiple SGNN architectures and real-world datasets demonstrate both the effectiveness of our proposed balance-attack and the superior robustness of BA-SGCL, advancing the security and reliability of signed graph analysis in social networks. Datasets and codes of the proposed framework are at the github repository https://anonymous.4open.science/r/BA-SGCL-submit-DF41/.



## **42. A Review of the Duality of Adversarial Learning in Network Intrusion: Attacks and Countermeasures**

cs.CR

23 pages, 2 figures, 5 tables

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13880v1) [paper-pdf](http://arxiv.org/pdf/2412.13880v1)

**Authors**: Shalini Saini, Anitha Chennamaneni, Babatunde Sawyerr

**Abstract**: Deep learning solutions are instrumental in cybersecurity, harnessing their ability to analyze vast datasets, identify complex patterns, and detect anomalies. However, malevolent actors can exploit these capabilities to orchestrate sophisticated attacks, posing significant challenges to defenders and traditional security measures. Adversarial attacks, particularly those targeting vulnerabilities in deep learning models, present a nuanced and substantial threat to cybersecurity. Our study delves into adversarial learning threats such as Data Poisoning, Test Time Evasion, and Reverse Engineering, specifically impacting Network Intrusion Detection Systems. Our research explores the intricacies and countermeasures of attacks to deepen understanding of network security challenges amidst adversarial threats. In our study, we present insights into the dynamic realm of adversarial learning and its implications for network intrusion. The intersection of adversarial attacks and defenses within network traffic data, coupled with advances in machine learning and deep learning techniques, represents a relatively underexplored domain. Our research lays the groundwork for strengthening defense mechanisms to address the potential breaches in network security and privacy posed by adversarial attacks. Through our in-depth analysis, we identify domain-specific research gaps, such as the scarcity of real-life attack data and the evaluation of AI-based solutions for network traffic. Our focus on these challenges aims to stimulate future research efforts toward the development of resilient network defense strategies.



## **43. Formal Verification of Permission Voucher**

cs.CR

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.16224v1) [paper-pdf](http://arxiv.org/pdf/2412.16224v1)

**Authors**: Khan Reaz, Gerhard Wunder

**Abstract**: Formal verification is a critical process in ensuring the security and correctness of cryptographic protocols, particularly in high-assurance domains. This paper presents a comprehensive formal analysis of the Permission Voucher Protocol, a system designed for secure and authenticated access control in distributed environments. The analysis employs the Tamarin Prover, a state-of-the-art tool for symbolic verification, to evaluate key security properties such as authentication, confidentiality, integrity, mutual authentication, and replay prevention. We model the protocol's components, including trust relationships, secure channels, and adversary capabilities under the Dolev-Yao model. Verification results confirm the protocol's robustness against common attacks such as message tampering, impersonation, and replay. Additionally, dependency graphs and detailed proofs demonstrate the successful enforcement of security properties like voucher authenticity, data confidentiality, and key integrity. The study identifies potential enhancements, such as incorporating timestamp-based validity checks and augmenting mutual authentication mechanisms to address insider threats and key management challenges. This work highlights the advantages and limitations of using the Tamarin Prover for formal security verification and proposes strategies to mitigate scalability and performance constraints in complex systems.



## **44. Cultivating Archipelago of Forests: Evolving Robust Decision Trees through Island Coevolution**

cs.LG

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13762v1) [paper-pdf](http://arxiv.org/pdf/2412.13762v1)

**Authors**: Adam Żychowski, Andrew Perrault, Jacek Mańdziuk

**Abstract**: Decision trees are widely used in machine learning due to their simplicity and interpretability, but they often lack robustness to adversarial attacks and data perturbations. The paper proposes a novel island-based coevolutionary algorithm (ICoEvoRDF) for constructing robust decision tree ensembles. The algorithm operates on multiple islands, each containing populations of decision trees and adversarial perturbations. The populations on each island evolve independently, with periodic migration of top-performing decision trees between islands. This approach fosters diversity and enhances the exploration of the solution space, leading to more robust and accurate decision tree ensembles. ICoEvoRDF utilizes a popular game theory concept of mixed Nash equilibrium for ensemble weighting, which further leads to improvement in results. ICoEvoRDF is evaluated on 20 benchmark datasets, demonstrating its superior performance compared to state-of-the-art methods in optimizing both adversarial accuracy and minimax regret. The flexibility of ICoEvoRDF allows for the integration of decision trees from various existing methods, providing a unified framework for combining diverse solutions. Our approach offers a promising direction for developing robust and interpretable machine learning models



## **45. A2RNet: Adversarial Attack Resilient Network for Robust Infrared and Visible Image Fusion**

cs.CV

9 pages, 8 figures, The 39th Annual AAAI Conference on Artificial  Intelligence

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.09954v2) [paper-pdf](http://arxiv.org/pdf/2412.09954v2)

**Authors**: Jiawei Li, Hongwei Yu, Jiansheng Chen, Xinlong Ding, Jinlong Wang, Jinyuan Liu, Bochao Zou, Huimin Ma

**Abstract**: Infrared and visible image fusion (IVIF) is a crucial technique for enhancing visual performance by integrating unique information from different modalities into one fused image. Exiting methods pay more attention to conducting fusion with undisturbed data, while overlooking the impact of deliberate interference on the effectiveness of fusion results. To investigate the robustness of fusion models, in this paper, we propose a novel adversarial attack resilient network, called $\textrm{A}^{\textrm{2}}$RNet. Specifically, we develop an adversarial paradigm with an anti-attack loss function to implement adversarial attacks and training. It is constructed based on the intrinsic nature of IVIF and provide a robust foundation for future research advancements. We adopt a Unet as the pipeline with a transformer-based defensive refinement module (DRM) under this paradigm, which guarantees fused image quality in a robust coarse-to-fine manner. Compared to previous works, our method mitigates the adverse effects of adversarial perturbations, consistently maintaining high-fidelity fusion results. Furthermore, the performance of downstream tasks can also be well maintained under adversarial attacks. Code is available at https://github.com/lok-18/A2RNet.



## **46. Physics-Based Adversarial Attack on Near-Infrared Human Detector for Nighttime Surveillance Camera Systems**

cs.CV

Appeared in ACM MM 2023

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13709v1) [paper-pdf](http://arxiv.org/pdf/2412.13709v1)

**Authors**: Muyao Niu, Zhuoxiao Li, Yifan Zhan, Huy H. Nguyen, Isao Echizen, Yinqiang Zheng

**Abstract**: Many surveillance cameras switch between daytime and nighttime modes based on illuminance levels. During the day, the camera records ordinary RGB images through an enabled IR-cut filter. At night, the filter is disabled to capture near-infrared (NIR) light emitted from NIR LEDs typically mounted around the lens. While RGB-based AI algorithm vulnerabilities have been widely reported, the vulnerabilities of NIR-based AI have rarely been investigated. In this paper, we identify fundamental vulnerabilities in NIR-based image understanding caused by color and texture loss due to the intrinsic characteristics of clothes' reflectance and cameras' spectral sensitivity in the NIR range. We further show that the nearly co-located configuration of illuminants and cameras in existing surveillance systems facilitates concealing and fully passive attacks in the physical world. Specifically, we demonstrate how retro-reflective and insulation plastic tapes can manipulate the intensity distribution of NIR images. We showcase an attack on the YOLO-based human detector using binary patterns designed in the digital space (via black-box query and searching) and then physically realized using tapes pasted onto clothes. Our attack highlights significant reliability concerns for nighttime surveillance systems, which are intended to enhance security. Codes Available: https://github.com/MyNiuuu/AdvNIR



## **47. Mitigating Adversarial Attacks in LLMs through Defensive Suffix Generation**

cs.CV

9 pages, 2 figures

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13705v1) [paper-pdf](http://arxiv.org/pdf/2412.13705v1)

**Authors**: Minkyoung Kim, Yunha Kim, Hyeram Seo, Heejung Choi, Jiye Han, Gaeun Kee, Soyoung Ko, HyoJe Jung, Byeolhee Kim, Young-Hak Kim, Sanghyun Park, Tae Joon Jun

**Abstract**: Large language models (LLMs) have exhibited outstanding performance in natural language processing tasks. However, these models remain susceptible to adversarial attacks in which slight input perturbations can lead to harmful or misleading outputs. A gradient-based defensive suffix generation algorithm is designed to bolster the robustness of LLMs. By appending carefully optimized defensive suffixes to input prompts, the algorithm mitigates adversarial influences while preserving the models' utility. To enhance adversarial understanding, a novel total loss function ($L_{\text{total}}$) combining defensive loss ($L_{\text{def}}$) and adversarial loss ($L_{\text{adv}}$) generates defensive suffixes more effectively. Experimental evaluations conducted on open-source LLMs such as Gemma-7B, mistral-7B, Llama2-7B, and Llama2-13B show that the proposed method reduces attack success rates (ASR) by an average of 11\% compared to models without defensive suffixes. Additionally, the perplexity score of Gemma-7B decreased from 6.57 to 3.93 when applying the defensive suffix generated by openELM-270M. Furthermore, TruthfulQA evaluations demonstrate consistent improvements with Truthfulness scores increasing by up to 10\% across tested configurations. This approach significantly enhances the security of LLMs in critical applications without requiring extensive retraining.



## **48. Enhancing Adversarial Transferability with Adversarial Weight Tuning**

cs.CR

Accepted by AAAI 2025

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2408.09469v3) [paper-pdf](http://arxiv.org/pdf/2408.09469v3)

**Authors**: Jiahao Chen, Zhou Feng, Rui Zeng, Yuwen Pu, Chunyi Zhou, Yi Jiang, Yuyou Gan, Jinbao Li, Shouling Ji

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples (AEs) that mislead the model while appearing benign to human observers. A critical concern is the transferability of AEs, which enables black-box attacks without direct access to the target model. However, many previous attacks have failed to explain the intrinsic mechanism of adversarial transferability. In this paper, we rethink the property of transferable AEs and reformalize the formulation of transferability. Building on insights from this mechanism, we analyze the generalization of AEs across models with different architectures and prove that we can find a local perturbation to mitigate the gap between surrogate and target models. We further establish the inner connections between model smoothness and flat local maxima, both of which contribute to the transferability of AEs. Further, we propose a new adversarial attack algorithm, \textbf{A}dversarial \textbf{W}eight \textbf{T}uning (AWT), which adaptively adjusts the parameters of the surrogate model using generated AEs to optimize the flat local maxima and model smoothness simultaneously, without the need for extra data. AWT is a data-free tuning method that combines gradient-based and model-based attack methods to enhance the transferability of AEs. Extensive experiments on a variety of models with different architectures on ImageNet demonstrate that AWT yields superior performance over other attacks, with an average increase of nearly 5\% and 10\% attack success rates on CNN-based and Transformer-based models, respectively, compared to state-of-the-art attacks.



## **49. Understanding Key Point Cloud Features for Development Three-dimensional Adversarial Attacks**

cs.CV

10 pages, 6 figures

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2210.14164v4) [paper-pdf](http://arxiv.org/pdf/2210.14164v4)

**Authors**: Hanieh Naderi, Chinthaka Dinesh, Ivan V. Bajic, Shohreh Kasaei

**Abstract**: Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of three-dimensional point clouds, methods have been developed to identify points that play a key role in network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. This paper seeks to enhance the understanding of three-dimensional adversarial attacks by exploring which point cloud features are most important for predicting adversarial points. Specifically, Fourteen key point cloud features such as edge intensity and distance from the centroid are defined, and multiple linear regression is employed to assess their predictive power for adversarial points. Based on critical feature selection insights, a new attack method has been developed to evaluate whether the selected features can generate an attack successfully. Unlike traditional attack methods that rely on model-specific vulnerabilities, this approach focuses on the intrinsic characteristics of the point clouds themselves. It is demonstrated that these features can predict adversarial points across four different DNN architectures, Point Network (PointNet), PointNet++, Dynamic Graph Convolutional Neural Networks (DGCNN), and Point Convolutional Network (PointConv) outperforming random guessing and achieving results comparable to saliency map-based attacks. This study has important engineering applications, such as enhancing the security and robustness of three-dimensional point cloud-based systems in fields like robotics and autonomous driving.



## **50. Novel AI Camera Camouflage: Face Cloaking Without Full Disguise**

cs.CV

**SubmitDate**: 2024-12-18    [abs](http://arxiv.org/abs/2412.13507v1) [paper-pdf](http://arxiv.org/pdf/2412.13507v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: This study demonstrates a novel approach to facial camouflage that combines targeted cosmetic perturbations and alpha transparency layer manipulation to evade modern facial recognition systems. Unlike previous methods -- such as CV dazzle, adversarial patches, and theatrical disguises -- this work achieves effective obfuscation through subtle modifications to key-point regions, particularly the brow, nose bridge, and jawline. Empirical testing with Haar cascade classifiers and commercial systems like BetaFaceAPI and Microsoft Bing Visual Search reveals that vertical perturbations near dense facial key points significantly disrupt detection without relying on overt disguises. Additionally, leveraging alpha transparency attacks in PNG images creates a dual-layer effect: faces remain visible to human observers but disappear in machine-readable RGB layers, rendering them unidentifiable during reverse image searches. The results highlight the potential for creating scalable, low-visibility facial obfuscation strategies that balance effectiveness and subtlety, opening pathways for defeating surveillance while maintaining plausible anonymity.



