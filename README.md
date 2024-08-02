# Latest Adversarial Attack Papers
**update at 2024-08-02 16:08:23**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Tamper-Resistant Safeguards for Open-Weight LLMs**

cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00761v1) [paper-pdf](http://arxiv.org/pdf/2408.00761v1)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.



## **2. CERT-ED: Certifiably Robust Text Classification for Edit Distance**

cs.CL

22 pages, 3 figures, 12 tables. Include 11 pages of appendices

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00728v1) [paper-pdf](http://arxiv.org/pdf/2408.00728v1)

**Authors**: Zhuoqun Huang, Neil G Marchant, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstract**: With the growing integration of AI in daily life, ensuring the robustness of systems to inference-time attacks is crucial. Among the approaches for certifying robustness to such adversarial examples, randomized smoothing has emerged as highly promising due to its nature as a wrapper around arbitrary black-box models. Previous work on randomized smoothing in natural language processing has primarily focused on specific subsets of edit distance operations, such as synonym substitution or word insertion, without exploring the certification of all edit operations. In this paper, we adapt Randomized Deletion (Huang et al., 2023) and propose, CERTified Edit Distance defense (CERT-ED) for natural language classification. Through comprehensive experiments, we demonstrate that CERT-ED outperforms the existing Hamming distance method RanMASK (Zeng et al., 2023) in 4 out of 5 datasets in terms of both accuracy and the cardinality of the certificate. By covering various threat models, including 5 direct and 5 transfer attacks, our method improves empirical robustness in 38 out of 50 settings.



## **3. Prover-Verifier Games improve legibility of LLM outputs**

cs.CL

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2407.13692v2) [paper-pdf](http://arxiv.org/pdf/2407.13692v2)

**Authors**: Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, Yuri Burda

**Abstract**: One way to increase confidence in the outputs of Large Language Models (LLMs) is to support them with reasoning that is clear and easy to check -- a property we call legibility. We study legibility in the context of solving grade-school math problems and show that optimizing chain-of-thought solutions only for answer correctness can make them less legible. To mitigate the loss in legibility, we propose a training algorithm inspired by Prover-Verifier Game from Anil et al. (2021). Our algorithm iteratively trains small verifiers to predict solution correctness, "helpful" provers to produce correct solutions that the verifier accepts, and "sneaky" provers to produce incorrect solutions that fool the verifier. We find that the helpful prover's accuracy and the verifier's robustness to adversarial attacks increase over the course of training. Furthermore, we show that legibility training transfers to time-constrained humans tasked with verifying solution correctness. Over course of LLM training human accuracy increases when checking the helpful prover's solutions, and decreases when checking the sneaky prover's solutions. Hence, training for checkability by small verifiers is a plausible technique for increasing output legibility. Our results suggest legibility training against small verifiers as a practical avenue for increasing legibility of large LLMs to humans, and thus could help with alignment of superhuman models.



## **4. Defending Jailbreak Attack in VLMs via Cross-modality Information Detector**

cs.CL

12 pages, 9 figures

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2407.21659v2) [paper-pdf](http://arxiv.org/pdf/2407.21659v2)

**Authors**: Yue Xu, Xiuyuan Qi, Zhan Qin, Wenjie Wang

**Abstract**: Vision Language Models (VLMs) extend the capacity of LLMs to comprehensively understand vision information, achieving remarkable performance in many vision-centric tasks. Despite that, recent studies have shown that these models are susceptible to jailbreak attacks, which refer to an exploitative technique where malicious users can break the safety alignment of the target model and generate misleading and harmful answers. This potential threat is caused by both the inherent vulnerabilities of LLM and the larger attack scope introduced by vision input. To enhance the security of VLMs against jailbreak attacks, researchers have developed various defense techniques. However, these methods either require modifications to the model's internal structure or demand significant computational resources during the inference phase. Multimodal information is a double-edged sword. While it increases the risk of attacks, it also provides additional data that can enhance safeguards. Inspired by this, we propose $\underline{\textbf{C}}$ross-modality $\underline{\textbf{I}}$nformation $\underline{\textbf{DE}}$tecto$\underline{\textbf{R}}$ ($\textit{CIDER})$, a plug-and-play jailbreaking detector designed to identify maliciously perturbed image inputs, utilizing the cross-modal similarity between harmful queries and adversarial images. This simple yet effective cross-modality information detector, $\textit{CIDER}$, is independent of the target VLMs and requires less computation cost. Extensive experimental results demonstrate the effectiveness and efficiency of $\textit{CIDER}$, as well as its transferability to both white-box and black-box VLMs.



## **5. Autonomous LLM-Enhanced Adversarial Attack for Text-to-Motion**

cs.CV

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00352v1) [paper-pdf](http://arxiv.org/pdf/2408.00352v1)

**Authors**: Honglei Miao, Fan Ma, Ruijie Quan, Kun Zhan, Yi Yang

**Abstract**: Human motion generation driven by deep generative models has enabled compelling applications, but the ability of text-to-motion (T2M) models to produce realistic motions from text prompts raises security concerns if exploited maliciously. Despite growing interest in T2M, few methods focus on safeguarding these models against adversarial attacks, with existing work on text-to-image models proving insufficient for the unique motion domain. In the paper, we propose ALERT-Motion, an autonomous framework leveraging large language models (LLMs) to craft targeted adversarial attacks against black-box T2M models. Unlike prior methods modifying prompts through predefined rules, ALERT-Motion uses LLMs' knowledge of human motion to autonomously generate subtle yet powerful adversarial text descriptions. It comprises two key modules: an adaptive dispatching module that constructs an LLM-based agent to iteratively refine and search for adversarial prompts; and a multimodal information contrastive module that extracts semantically relevant motion information to guide the agent's search. Through this LLM-driven approach, ALERT-Motion crafts adversarial prompts querying victim models to produce outputs closely matching targeted motions, while avoiding obvious perturbations. Evaluations across popular T2M models demonstrate ALERT-Motion's superiority over previous methods, achieving higher attack success rates with stealthier adversarial prompts. This pioneering work on T2M adversarial attacks highlights the urgency of developing defensive measures as motion generation technology advances, urging further research into safe and responsible deployment.



## **6. Securing the Diagnosis of Medical Imaging: An In-depth Analysis of AI-Resistant Attacks**

cs.CR

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00348v1) [paper-pdf](http://arxiv.org/pdf/2408.00348v1)

**Authors**: Angona Biswas, MD Abdullah Al Nasim, Kishor Datta Gupta, Roy George, Abdur Rashid

**Abstract**: Machine learning (ML) is a rapidly developing area of medicine that uses significant resources to apply computer science and statistics to medical issues. ML's proponents laud its capacity to handle vast, complicated, and erratic medical data. It's common knowledge that attackers might cause misclassification by deliberately creating inputs for machine learning classifiers. Research on adversarial examples has been extensively conducted in the field of computer vision applications. Healthcare systems are thought to be highly difficult because of the security and life-or-death considerations they include, and performance accuracy is very important. Recent arguments have suggested that adversarial attacks could be made against medical image analysis (MedIA) technologies because of the accompanying technology infrastructure and powerful financial incentives. Since the diagnosis will be the basis for important decisions, it is essential to assess how strong medical DNN tasks are against adversarial attacks. Simple adversarial attacks have been taken into account in several earlier studies. However, DNNs are susceptible to more risky and realistic attacks. The present paper covers recent proposed adversarial attack strategies against DNNs for medical imaging as well as countermeasures. In this study, we review current techniques for adversarial imaging attacks, detections. It also encompasses various facets of these techniques and offers suggestions for the robustness of neural networks to be improved in the future.



## **7. MAARS: Multi-Rate Attack-Aware Randomized Scheduling for Securing Real-time Systems**

eess.SY

12 pages including references, Total 10 figures (with 3 having  subfigures). This paper was rejected in RTSS 2024 Conference

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00341v1) [paper-pdf](http://arxiv.org/pdf/2408.00341v1)

**Authors**: Arkaprava Sain, Sunandan Adhikary, Ipsita Koley, Soumyajit Dey

**Abstract**: Modern Cyber-Physical Systems (CPSs) consist of numerous control units interconnected by communication networks. Each control unit executes multiple safety-critical and non-critical tasks in real-time. Most of the safety-critical tasks are executed with a fixed sampling period to ensure deterministic timing behaviour that helps in its safety and performance analysis. However, adversaries can exploit this deterministic behaviour of safety-critical tasks to launch inference-based-based attacks on them. This paper aims to prevent and minimize the possibility of such timing inference or schedule-based attacks to compromise the control units. This is done by switching between strategically chosen execution rates of the safety-critical control tasks such that their performance remains unhampered. Thereafter, we present a novel schedule vulnerability analysis methodology to switch between valid schedules generated for these multiple periodicities of the control tasks in run time. Utilizing these strategies, we introduce a novel Multi-Rate Attack-Aware Randomized Scheduling (MAARS) framework for preemptive fixed-priority schedulers that minimize the success rate of timing-inference-based attacks on safety-critical real-time systems. To our knowledge, this is the first work to propose a schedule randomization method with attack awareness that preserves both the control and scheduling aspects. The efficacy of the framework in terms of attack prevention is finally evaluated on several automotive benchmarks in a Hardware-in-loop (HiL) environment.



## **8. OTAD: An Optimal Transport-Induced Robust Model for Agnostic Adversarial Attack**

cs.LG

14 pages, 2 figures

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00329v1) [paper-pdf](http://arxiv.org/pdf/2408.00329v1)

**Authors**: Kuo Gai, Sicong Wang, Shihua Zhang

**Abstract**: Deep neural networks (DNNs) are vulnerable to small adversarial perturbations of the inputs, posing a significant challenge to their reliability and robustness. Empirical methods such as adversarial training can defend against particular attacks but remain vulnerable to more powerful attacks. Alternatively, Lipschitz networks provide certified robustness to unseen perturbations but lack sufficient expressive power. To harness the advantages of both approaches, we design a novel two-step Optimal Transport induced Adversarial Defense (OTAD) model that can fit the training data accurately while preserving the local Lipschitz continuity. First, we train a DNN with a regularizer derived from optimal transport theory, yielding a discrete optimal transport map linking data to its features. By leveraging the map's inherent regularity, we interpolate the map by solving the convex integration problem (CIP) to guarantee the local Lipschitz property. OTAD is extensible to diverse architectures of ResNet and Transformer, making it suitable for complex data. For efficient computation, the CIP can be solved through training neural networks. OTAD opens a novel avenue for developing reliable and secure deep learning systems through the regularity of optimal transport maps. Empirical results demonstrate that OTAD can outperform other robust models on diverse datasets.



## **9. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

cs.LG

20 pages

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00315v1) [paper-pdf](http://arxiv.org/pdf/2408.00315v1)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.



## **10. Adversarial Text Rewriting for Text-aware Recommender Systems**

cs.IR

Accepted for publication at: 33rd ACM International Conference on  Information and Knowledge Management (CIKM 2024). Code and data at:  https://github.com/sejoonoh/ATR

**SubmitDate**: 2024-08-01    [abs](http://arxiv.org/abs/2408.00312v1) [paper-pdf](http://arxiv.org/pdf/2408.00312v1)

**Authors**: Sejoon Oh, Gaurav Verma, Srijan Kumar

**Abstract**: Text-aware recommender systems incorporate rich textual features, such as titles and descriptions, to generate item recommendations for users. The use of textual features helps mitigate cold-start problems, and thus, such recommender systems have attracted increased attention. However, we argue that the dependency on item descriptions makes the recommender system vulnerable to manipulation by adversarial sellers on e-commerce platforms. In this paper, we explore the possibility of such manipulation by proposing a new text rewriting framework to attack text-aware recommender systems. We show that the rewriting attack can be exploited by sellers to unfairly uprank their products, even though the adversarially rewritten descriptions are perceived as realistic by human evaluators. Methodologically, we investigate two different variations to carry out text rewriting attacks: (1) two-phase fine-tuning for greater attack performance, and (2) in-context learning for higher text rewriting quality. Experiments spanning 3 different datasets and 4 existing approaches demonstrate that recommender systems exhibit vulnerability against the proposed text rewriting attack. Our work adds to the existing literature around the robustness of recommender systems, while highlighting a new dimension of vulnerability in the age of large-scale automated text generation.



## **11. Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks**

cs.LG

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2212.10717v2) [paper-pdf](http://arxiv.org/pdf/2212.10717v2)

**Authors**: Jimmy Z. Di, Jack Douglas, Jayadev Acharya, Gautam Kamath, Ayush Sekhari

**Abstract**: We introduce camouflaged data poisoning attacks, a new attack vector that arises in the context of machine unlearning and other settings when model retraining may be induced. An adversary first adds a few carefully crafted points to the training dataset such that the impact on the model's predictions is minimal. The adversary subsequently triggers a request to remove a subset of the introduced points at which point the attack is unleashed and the model's predictions are negatively affected. In particular, we consider clean-label targeted attacks (in which the goal is to cause the model to misclassify a specific test point) on datasets including CIFAR-10, Imagenette, and Imagewoof. This attack is realized by constructing camouflage datapoints that mask the effect of a poisoned dataset.



## **12. Vera Verto: Multimodal Hijacking Attack**

cs.CR

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2408.00129v1) [paper-pdf](http://arxiv.org/pdf/2408.00129v1)

**Authors**: Minxing Zhang, Ahmed Salem, Michael Backes, Yang Zhang

**Abstract**: The increasing cost of training machine learning (ML) models has led to the inclusion of new parties to the training pipeline, such as users who contribute training data and companies that provide computing resources. This involvement of such new parties in the ML training process has introduced new attack surfaces for an adversary to exploit. A recent attack in this domain is the model hijacking attack, whereby an adversary hijacks a victim model to implement their own -- possibly malicious -- hijacking tasks. However, the scope of the model hijacking attack is so far limited to the homogeneous-modality tasks. In this paper, we transform the model hijacking attack into a more general multimodal setting, where the hijacking and original tasks are performed on data of different modalities. Specifically, we focus on the setting where an adversary implements a natural language processing (NLP) hijacking task into an image classification model. To mount the attack, we propose a novel encoder-decoder based framework, namely the Blender, which relies on advanced image and language models. Experimental results show that our modal hijacking attack achieves strong performances in different settings. For instance, our attack achieves 94%, 94%, and 95% attack success rate when using the Sogou news dataset to hijack STL10, CIFAR-10, and MNIST classifiers.



## **13. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

cs.CL

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2407.21630v1) [paper-pdf](http://arxiv.org/pdf/2407.21630v1)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduce the accuracy of attackers while preserving utility. We make our code and models publicly available.



## **14. On the Perturbed States for Transformed Input-robust Reinforcement Learning**

cs.LG

12 pages

**SubmitDate**: 2024-07-31    [abs](http://arxiv.org/abs/2408.00023v1) [paper-pdf](http://arxiv.org/pdf/2408.00023v1)

**Authors**: Tung M. Luu, Haeyong Kang, Tri Ton, Thanh Nguyen, Chang D. Yoo

**Abstract**: Reinforcement Learning (RL) agents demonstrating proficiency in a training environment exhibit vulnerability to adversarial perturbations in input observations during deployment. This underscores the importance of building a robust agent before its real-world deployment. To alleviate the challenging point, prior works focus on developing robust training-based procedures, encompassing efforts to fortify the deep neural network component's robustness or subject the agent to adversarial training against potent attacks. In this work, we propose a novel method referred to as \textit{Transformed Input-robust RL (TIRL)}, which explores another avenue to mitigate the impact of adversaries by employing input transformation-based defenses. Specifically, we introduce two principles for applying transformation-based defenses in learning robust RL agents: \textit{(1) autoencoder-styled denoising} to reconstruct the original state and \textit{(2) bounded transformations (bit-depth reduction and vector quantization (VQ))} to achieve close transformed inputs. The transformations are applied to the state before feeding it into the policy network. Extensive experiments on multiple \mujoco environments demonstrate that input transformation-based defenses, \ie, VQ, defend against several adversaries in the state observations.



## **15. DeepBaR: Fault Backdoor Attack on Deep Neural Network Layers**

cs.LG

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.21220v1) [paper-pdf](http://arxiv.org/pdf/2407.21220v1)

**Authors**: C. A. Martínez-Mejía, J. Solano, J. Breier, D. Bucko, X. Hou

**Abstract**: Machine Learning using neural networks has received prominent attention recently because of its success in solving a wide variety of computational tasks, in particular in the field of computer vision. However, several works have drawn attention to potential security risks involved with the training and implementation of such networks. In this work, we introduce DeepBaR, a novel approach that implants backdoors on neural networks by faulting their behavior at training, especially during fine-tuning. Our technique aims to generate adversarial samples by optimizing a custom loss function that mimics the implanted backdoors while adding an almost non-visible trigger in the image. We attack three popular convolutional neural network architectures and show that DeepBaR attacks have a success rate of up to 98.30\%. Furthermore, DeepBaR does not significantly affect the accuracy of the attacked networks after deployment when non-malicious inputs are given. Remarkably, DeepBaR allows attackers to choose an input that looks similar to a given class, from a human perspective, but that will be classified as belonging to an arbitrary target class.



## **16. AI Safety in Practice: Enhancing Adversarial Robustness in Multimodal Image Captioning**

cs.CV

Accepted into KDD 2024 workshop on Ethical AI

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.21174v1) [paper-pdf](http://arxiv.org/pdf/2407.21174v1)

**Authors**: Maisha Binte Rashid, Pablo Rivas

**Abstract**: Multimodal machine learning models that combine visual and textual data are increasingly being deployed in critical applications, raising significant safety and security concerns due to their vulnerability to adversarial attacks. This paper presents an effective strategy to enhance the robustness of multimodal image captioning models against such attacks. By leveraging the Fast Gradient Sign Method (FGSM) to generate adversarial examples and incorporating adversarial training techniques, we demonstrate improved model robustness on two benchmark datasets: Flickr8k and COCO. Our findings indicate that selectively training only the text decoder of the multimodal architecture shows performance comparable to full adversarial training while offering increased computational efficiency. This targeted approach suggests a balance between robustness and training costs, facilitating the ethical deployment of multimodal AI systems across various domains.



## **17. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

cs.CV

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20836v1) [paper-pdf](http://arxiv.org/pdf/2407.20836v1)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of these AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. For the task of AIGI detection, we propose a new attack containing two main parts. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous models, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as frequency-based post-train Bayesian attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario.



## **18. Attacking Cooperative Multi-Agent Reinforcement Learning by Adversarial Minority Influence**

cs.LG

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2302.03322v3) [paper-pdf](http://arxiv.org/pdf/2302.03322v3)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Yuwei Zheng, Pu Feng, Xin Yu, Aishan Liu, Yaodong Yang, Bo An, Wenjun Wu, Xianglong Liu

**Abstract**: This study probes the vulnerabilities of cooperative multi-agent reinforcement learning (c-MARL) under adversarial attacks, a critical determinant of c-MARL's worst-case performance prior to real-world implementation. Current observation-based attacks, constrained by white-box assumptions, overlook c-MARL's complex multi-agent interactions and cooperative objectives, resulting in impractical and limited attack capabilities. To address these shortcomes, we propose Adversarial Minority Influence (AMI), a practical and strong for c-MARL. AMI is a practical black-box attack and can be launched without knowing victim parameters. AMI is also strong by considering the complex multi-agent interaction and the cooperative goal of agents, enabling a single adversarial agent to unilaterally misleads majority victims to form targeted worst-case cooperation. This mirrors minority influence phenomena in social psychology. To achieve maximum deviation in victim policies under complex agent-wise interactions, our unilateral attack aims to characterize and maximize the impact of the adversary on the victims. This is achieved by adapting a unilateral agent-wise relation metric derived from mutual information, thereby mitigating the adverse effects of victim influence on the adversary. To lead the victims into a jointly detrimental scenario, our targeted attack deceives victims into a long-term, cooperatively harmful situation by guiding each victim towards a specific target, determined through a trial-and-error process executed by a reinforcement learning agent. Through AMI, we achieve the first successful attack against real-world robot swarms and effectively fool agents in simulated environments into collectively worst-case scenarios, including Starcraft II and Multi-agent Mujoco. The source code and demonstrations can be found at: https://github.com/DIG-Beihang/AMI.



## **19. Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks**

cs.CV

Accepted to ECCV 2024, Project Page: https://PDCL-Attack.github.io

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20657v1) [paper-pdf](http://arxiv.org/pdf/2407.20657v1)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Recent vision-language foundation models, such as CLIP, have demonstrated superior capabilities in learning representations that can be transferable across diverse range of downstream tasks and domains. With the emergence of such powerful models, it has become crucial to effectively leverage their capabilities in tackling challenging vision tasks. On the other hand, only a few works have focused on devising adversarial examples that transfer well to both unknown domains and model architectures. In this paper, we propose a novel transfer attack method called PDCL-Attack, which leverages the CLIP model to enhance the transferability of adversarial perturbations generated by a generative model-based attack framework. Specifically, we formulate an effective prompt-driven feature guidance by harnessing the semantic representation power of text, particularly from the ground-truth class labels of input images. To the best of our knowledge, we are the first to introduce prompt learning to enhance the transferable generative attacks. Extensive experiments conducted across various cross-domain and cross-model settings empirically validate our approach, demonstrating its superiority over state-of-the-art methods.



## **20. FACL-Attack: Frequency-Aware Contrastive Learning for Transferable Adversarial Attacks**

cs.CV

Accepted to AAAI 2024, Project Page: https://FACL-Attack.github.io

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2407.20653v1) [paper-pdf](http://arxiv.org/pdf/2407.20653v1)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Deep neural networks are known to be vulnerable to security risks due to the inherent transferable nature of adversarial examples. Despite the success of recent generative model-based attacks demonstrating strong transferability, it still remains a challenge to design an efficient attack strategy in a real-world strict black-box setting, where both the target domain and model architectures are unknown. In this paper, we seek to explore a feature contrastive approach in the frequency domain to generate adversarial examples that are robust in both cross-domain and cross-model settings. With that goal in mind, we propose two modules that are only employed during the training phase: a Frequency-Aware Domain Randomization (FADR) module to randomize domain-variant low- and high-range frequency components and a Frequency-Augmented Contrastive Learning (FACL) module to effectively separate domain-invariant mid-frequency features of clean and perturbed image. We demonstrate strong transferability of our generated adversarial perturbations through extensive cross-domain and cross-model experiments, while keeping the inference time complexity.



## **21. Robust Federated Learning for Wireless Networks: A Demonstration with Channel Estimation**

cs.LG

Submitted to IEEE GLOBECOM 2024

**SubmitDate**: 2024-07-30    [abs](http://arxiv.org/abs/2404.03088v2) [paper-pdf](http://arxiv.org/pdf/2404.03088v2)

**Authors**: Zexin Fang, Bin Han, Hans D. Schotten

**Abstract**: Federated learning (FL) offers a privacy-preserving collaborative approach for training models in wireless networks, with channel estimation emerging as a promising application. Despite extensive studies on FL-empowered channel estimation, the security concerns associated with FL require meticulous attention. In a scenario where small base stations (SBSs) serve as local models trained on cached data, and a macro base station (MBS) functions as the global model setting, an attacker can exploit the vulnerability of FL, launching attacks with various adversarial attacks or deployment tactics. In this paper, we analyze such vulnerabilities, corresponding solutions were brought forth, and validated through simulation.



## **22. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20361v1) [paper-pdf](http://arxiv.org/pdf/2407.20361v1)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of two existing models, Stack model and Phishpedia, in classifying PhishOracle-generated adversarial phishing webpages. Additionally, we study a commercial large language model, Gemini Pro Vision, in the context of adversarial attacks. We conduct a user study to determine whether PhishOracle-generated adversarial phishing webpages deceive users. Our findings reveal that many PhishOracle-generated phishing webpages evade current phishing webpage detection models and deceive users, but Gemini Pro Vision is robust to the attack. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources are publicly available on GitHub.



## **23. Prompt Leakage effect and defense strategies for multi-turn LLM interactions**

cs.CR

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2404.16251v3) [paper-pdf](http://arxiv.org/pdf/2404.16251v3)

**Authors**: Divyansh Agarwal, Alexander R. Fabbri, Ben Risher, Philippe Laban, Shafiq Joty, Chien-Sheng Wu

**Abstract**: Prompt leakage poses a compelling security and privacy threat in LLM applications. Leakage of system prompts may compromise intellectual property, and act as adversarial reconnaissance for an attacker. A systematic evaluation of prompt leakage threats and mitigation strategies is lacking, especially for multi-turn LLM interactions. In this paper, we systematically investigate LLM vulnerabilities against prompt leakage for 10 closed- and open-source LLMs, across four domains. We design a unique threat model which leverages the LLM sycophancy effect and elevates the average attack success rate (ASR) from 17.7% to 86.2% in a multi-turn setting. Our standardized setup further allows dissecting leakage of specific prompt contents such as task instructions and knowledge documents. We measure the mitigation effect of 7 black-box defense strategies, along with finetuning an open-source model to defend against leakage attempts. We present different combination of defenses against our threat model, including a cost analysis. Our study highlights key takeaways for building secure LLM applications and provides directions for research in multi-turn LLM interactions



## **24. DDAP: Dual-Domain Anti-Personalization against Text-to-Image Diffusion Models**

cs.CV

Accepted by IJCB 2024

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.20141v1) [paper-pdf](http://arxiv.org/pdf/2407.20141v1)

**Authors**: Jing Yang, Runping Xi, Yingxin Lai, Xun Lin, Zitong Yu

**Abstract**: Diffusion-based personalized visual content generation technologies have achieved significant breakthroughs, allowing for the creation of specific objects by just learning from a few reference photos. However, when misused to fabricate fake news or unsettling content targeting individuals, these technologies could cause considerable societal harm. To address this problem, current methods generate adversarial samples by adversarially maximizing the training loss, thereby disrupting the output of any personalized generation model trained with these samples. However, the existing methods fail to achieve effective defense and maintain stealthiness, as they overlook the intrinsic properties of diffusion models. In this paper, we introduce a novel Dual-Domain Anti-Personalization framework (DDAP). Specifically, we have developed Spatial Perturbation Learning (SPL) by exploiting the fixed and perturbation-sensitive nature of the image encoder in personalized generation. Subsequently, we have designed a Frequency Perturbation Learning (FPL) method that utilizes the characteristics of diffusion models in the frequency domain. The SPL disrupts the overall texture of the generated images, while the FPL focuses on image details. By alternating between these two methods, we construct the DDAP framework, effectively harnessing the strengths of both domains. To further enhance the visual quality of the adversarial samples, we design a localization module to accurately capture attentive areas while ensuring the effectiveness of the attack and avoiding unnecessary disturbances in the background. Extensive experiments on facial benchmarks have shown that the proposed DDAP enhances the disruption of personalized generation models while also maintaining high quality in adversarial samples, making it more effective in protecting privacy in practical applications.



## **25. Adversarial Robustness in RGB-Skeleton Action Recognition: Leveraging Attention Modality Reweighter**

cs.CV

Accepted by IJCB 2024

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.19981v1) [paper-pdf](http://arxiv.org/pdf/2407.19981v1)

**Authors**: Chao Liu, Xin Liu, Zitong Yu, Yonghong Hou, Huanjing Yue, Jingyu Yang

**Abstract**: Deep neural networks (DNNs) have been applied in many computer vision tasks and achieved state-of-the-art (SOTA) performance. However, misclassification will occur when DNNs predict adversarial examples which are created by adding human-imperceptible adversarial noise to natural examples. This limits the application of DNN in security-critical fields. In order to enhance the robustness of models, previous research has primarily focused on the unimodal domain, such as image recognition and video understanding. Although multi-modal learning has achieved advanced performance in various tasks, such as action recognition, research on the robustness of RGB-skeleton action recognition models is scarce. In this paper, we systematically investigate how to improve the robustness of RGB-skeleton action recognition models. We initially conducted empirical analysis on the robustness of different modalities and observed that the skeleton modality is more robust than the RGB modality. Motivated by this observation, we propose the \formatword{A}ttention-based \formatword{M}odality \formatword{R}eweighter (\formatword{AMR}), which utilizes an attention layer to re-weight the two modalities, enabling the model to learn more robust features. Our AMR is plug-and-play, allowing easy integration with multimodal models. To demonstrate the effectiveness of AMR, we conducted extensive experiments on various datasets. For example, compared to the SOTA methods, AMR exhibits a 43.77\% improvement against PGD20 attacks on the NTU-RGB+D 60 dataset. Furthermore, it effectively balances the differences in robustness between different modalities.



## **26. Quasi-Framelets: Robust Graph Neural Networks via Adaptive Framelet Convolution**

cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2201.04728v2) [paper-pdf](http://arxiv.org/pdf/2201.04728v2)

**Authors**: Mengxi Yang, Dai Shi, Xuebin Zheng, Jie Yin, Junbin Gao

**Abstract**: This paper aims to provide a novel design of a multiscale framelet convolution for spectral graph neural networks (GNNs). While current spectral methods excel in various graph learning tasks, they often lack the flexibility to adapt to noisy, incomplete, or perturbed graph signals, making them fragile in such conditions. Our newly proposed framelet convolution addresses these limitations by decomposing graph data into low-pass and high-pass spectra through a finely-tuned multiscale approach. Our approach directly designs filtering functions within the spectral domain, allowing for precise control over the spectral components. The proposed design excels in filtering out unwanted spectral information and significantly reduces the adverse effects of noisy graph signals. Our approach not only enhances the robustness of GNNs but also preserves crucial graph features and structures. Through extensive experiments on diverse, real-world graph datasets, we demonstrate that our framelet convolution achieves superior performance in node classification tasks. It exhibits remarkable resilience to noisy data and adversarial attacks, highlighting its potential as a robust solution for real-world graph applications. This advancement opens new avenues for more adaptive and reliable spectral GNN architectures.



## **27. Detecting and Understanding Vulnerabilities in Language Models via Mechanistic Interpretability**

cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.19842v1) [paper-pdf](http://arxiv.org/pdf/2407.19842v1)

**Authors**: Jorge García-Carrasco, Alejandro Maté, Juan Trujillo

**Abstract**: Large Language Models (LLMs), characterized by being trained on broad amounts of data in a self-supervised manner, have shown impressive performance across a wide range of tasks. Indeed, their generative abilities have aroused interest on the application of LLMs across a wide range of contexts. However, neural networks in general, and LLMs in particular, are known to be vulnerable to adversarial attacks, where an imperceptible change to the input can mislead the output of the model. This is a serious concern that impedes the use of LLMs on high-stakes applications, such as healthcare, where a wrong prediction can imply serious consequences. Even though there are many efforts on making LLMs more robust to adversarial attacks, there are almost no works that study \emph{how} and \emph{where} these vulnerabilities that make LLMs prone to adversarial attacks happen. Motivated by these facts, we explore how to localize and understand vulnerabilities, and propose a method, based on Mechanistic Interpretability (MI) techniques, to guide this process. Specifically, this method enables us to detect vulnerabilities related to a concrete task by (i) obtaining the subset of the model that is responsible for that task, (ii) generating adversarial samples for that task, and (iii) using MI techniques together with the previous samples to discover and understand the possible vulnerabilities. We showcase our method on a pretrained GPT-2 Small model carrying out the task of predicting 3-letter acronyms to demonstrate its effectiveness on locating and understanding concrete vulnerabilities of the model.



## **28. Enhancing Adversarial Text Attacks on BERT Models with Projected Gradient Descent**

cs.LG

This paper is the pre-reviewed version of our paper that has been  accepted for oral presentation and publication in the 4th IEEE ASIANCON. The  conference will be organized in Pune, INDIA from August 23 to 25, 2024. The  paper consists of 8 pages and it contains 10 tables. It is NOT the final  camera-ready version that will be in IEEE Xplore

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2407.21073v1) [paper-pdf](http://arxiv.org/pdf/2407.21073v1)

**Authors**: Hetvi Waghela, Jaydip Sen, Sneha Rakshit

**Abstract**: Adversarial attacks against deep learning models represent a major threat to the security and reliability of natural language processing (NLP) systems. In this paper, we propose a modification to the BERT-Attack framework, integrating Projected Gradient Descent (PGD) to enhance its effectiveness and robustness. The original BERT-Attack, designed for generating adversarial examples against BERT-based models, suffers from limitations such as a fixed perturbation budget and a lack of consideration for semantic similarity. The proposed approach in this work, PGD-BERT-Attack, addresses these limitations by leveraging PGD to iteratively generate adversarial examples while ensuring both imperceptibility and semantic similarity to the original input. Extensive experiments are conducted to evaluate the performance of PGD-BERT-Attack compared to the original BERT-Attack and other baseline methods. The results demonstrate that PGD-BERT-Attack achieves higher success rates in causing misclassification while maintaining low perceptual changes. Furthermore, PGD-BERT-Attack produces adversarial instances that exhibit greater semantic resemblance to the initial input, enhancing their applicability in real-world scenarios. Overall, the proposed modification offers a more effective and robust approach to adversarial attacks on BERT-based models, thus contributing to the advancement of defense against attacks on NLP systems.



## **29. Understanding Robust Overfitting from the Feature Generalization Perspective**

cs.LG

**SubmitDate**: 2024-07-29    [abs](http://arxiv.org/abs/2310.00607v2) [paper-pdf](http://arxiv.org/pdf/2310.00607v2)

**Authors**: Chaojian Yu, Xiaolong Shi, Jun Yu, Bo Han, Tongliang Liu

**Abstract**: Adversarial training (AT) constructs robust neural networks by incorporating adversarial perturbations into natural data. However, it is plagued by the issue of robust overfitting (RO), which severely damages the model's robustness. In this paper, we investigate RO from a novel feature generalization perspective. Specifically, we design factor ablation experiments to assess the respective impacts of natural data and adversarial perturbations on RO, identifying that the inducing factor of RO stems from natural data. Given that the only difference between adversarial and natural training lies in the inclusion of adversarial perturbations, we further hypothesize that adversarial perturbations degrade the generalization of features in natural data and verify this hypothesis through extensive experiments. Based on these findings, we provide a holistic view of RO from the feature generalization perspective and explain various empirical behaviors associated with RO. To examine our feature generalization perspective, we devise two representative methods, attack strength and data augmentation, to prevent the feature generalization degradation during AT. Extensive experiments conducted on benchmark datasets demonstrate that the proposed methods can effectively mitigate RO and enhance adversarial robustness.



## **30. Exploring the Adversarial Robustness of CLIP for AI-generated Image Detection**

cs.CV

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.19553v1) [paper-pdf](http://arxiv.org/pdf/2407.19553v1)

**Authors**: Vincenzo De Rosa, Fabrizio Guillaro, Giovanni Poggi, Davide Cozzolino, Luisa Verdoliva

**Abstract**: In recent years, many forensic detectors have been proposed to detect AI-generated images and prevent their use for malicious purposes. Convolutional neural networks (CNNs) have long been the dominant architecture in this field and have been the subject of intense study. However, recently proposed Transformer-based detectors have been shown to match or even outperform CNN-based detectors, especially in terms of generalization. In this paper, we study the adversarial robustness of AI-generated image detectors, focusing on Contrastive Language-Image Pretraining (CLIP)-based methods that rely on Visual Transformer backbones and comparing their performance with CNN-based methods. We study the robustness to different adversarial attacks under a variety of conditions and analyze both numerical results and frequency-domain patterns. CLIP-based detectors are found to be vulnerable to white-box attacks just like CNN-based detectors. However, attacks do not easily transfer between CNN-based and CLIP-based methods. This is also confirmed by the different distribution of the adversarial noise patterns in the frequency domain. Overall, this analysis provides new insights into the properties of forensic detectors that can help to develop more effective strategies.



## **31. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

cs.LG

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.12068v2) [paper-pdf](http://arxiv.org/pdf/2407.12068v2)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.



## **32. Breaking the Balance of Power: Commitment Attacks on Ethereum's Reward Mechanism**

cs.CR

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2407.19479v1) [paper-pdf](http://arxiv.org/pdf/2407.19479v1)

**Authors**: Roozbeh Sarenche, Ertem Nusret Tas, Barnabe Monnot, Caspar Schwarz-Schilling, Bart Preneel

**Abstract**: Validators in permissionless, large-scale blockchains (e.g., Ethereum) are typically payoff-maximizing, rational actors. Ethereum relies on in-protocol incentives, like rewards for validators delivering correct and timely votes, to induce honest behavior and secure the blockchain. However, external incentives, such as the block proposer's opportunity to capture maximal extractable value (MEV), may tempt validators to deviate from honest protocol participation.   We show a series of commitment attacks on LMD GHOST, a core part of Ethereum's consensus mechanism. We demonstrate how a single adversarial block proposer can orchestrate long-range chain reorganizations by manipulating Ethereum's reward system for timely votes. These attacks disrupt the intended balance of power between proposers and voters: by leveraging credible threats, the adversarial proposer can coerce voters from previous slots into supporting blocks that conflict with the honest chain, enabling a chain reorganization at no cost to the adversary. In response, we introduce a novel reward mechanism that restores the voters' role as a check against proposer power. Our proposed mitigation is fairer and more decentralized -- not only in the context of these attacks -- but also practical for implementation in Ethereum.



## **33. Predominant Aspects on Security for Quantum Machine Learning: Literature Review**

quant-ph

Accepted at the IEEE International Conference on Quantum Computing  and Engineering (QCE)

**SubmitDate**: 2024-07-28    [abs](http://arxiv.org/abs/2401.07774v3) [paper-pdf](http://arxiv.org/pdf/2401.07774v3)

**Authors**: Nicola Franco, Alona Sakhnenko, Leon Stolpmann, Daniel Thuerck, Fabian Petsch, Annika Rüll, Jeanette Miriam Lorenz

**Abstract**: Quantum Machine Learning (QML) has emerged as a promising intersection of quantum computing and classical machine learning, anticipated to drive breakthroughs in computational tasks. This paper discusses the question which security concerns and strengths are connected to QML by means of a systematic literature review. We categorize and review the security of QML models, their vulnerabilities inherent to quantum architectures, and the mitigation strategies proposed. The survey reveals that while QML possesses unique strengths, it also introduces novel attack vectors not seen in classical systems. We point out specific risks, such as cross-talk in superconducting systems and forced repeated shuttle operations in ion-trap systems, which threaten QML's reliability. However, approaches like adversarial training, quantum noise exploitation, and quantum differential privacy have shown potential in enhancing QML robustness. Our review discuss the need for continued and rigorous research to ensure the secure deployment of QML in real-world applications. This work serves as a foundational reference for researchers and practitioners aiming to navigate the security aspects of QML.



## **34. A Closer Look at GAN Priors: Exploiting Intermediate Features for Enhanced Model Inversion Attacks**

cs.CV

ECCV 2024

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.13863v3) [paper-pdf](http://arxiv.org/pdf/2407.13863v3)

**Authors**: Yixiang Qiu, Hao Fang, Hongyao Yu, Bin Chen, MeiKang Qiu, Shu-Tao Xia

**Abstract**: Model Inversion (MI) attacks aim to reconstruct privacy-sensitive training data from released models by utilizing output information, raising extensive concerns about the security of Deep Neural Networks (DNNs). Recent advances in generative adversarial networks (GANs) have contributed significantly to the improved performance of MI attacks due to their powerful ability to generate realistic images with high fidelity and appropriate semantics. However, previous MI attacks have solely disclosed private information in the latent space of GAN priors, limiting their semantic extraction and transferability across multiple target models and datasets. To address this challenge, we propose a novel method, Intermediate Features enhanced Generative Model Inversion (IF-GMI), which disassembles the GAN structure and exploits features between intermediate blocks. This allows us to extend the optimization space from latent code to intermediate features with enhanced expressive capabilities. To prevent GAN priors from generating unrealistic images, we apply a L1 ball constraint to the optimization process. Experiments on multiple benchmarks demonstrate that our method significantly outperforms previous approaches and achieves state-of-the-art results under various settings, especially in the out-of-distribution (OOD) scenario. Our code is available at: https://github.com/final-solution/IF-GMI



## **35. EaTVul: ChatGPT-based Evasion Attack Against Software Vulnerability Detection**

cs.CR

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19216v1) [paper-pdf](http://arxiv.org/pdf/2407.19216v1)

**Authors**: Shigang Liu, Di Cao, Junae Kim, Tamas Abraham, Paul Montague, Seyit Camtepe, Jun Zhang, Yang Xiang

**Abstract**: Recently, deep learning has demonstrated promising results in enhancing the accuracy of vulnerability detection and identifying vulnerabilities in software. However, these techniques are still vulnerable to attacks. Adversarial examples can exploit vulnerabilities within deep neural networks, posing a significant threat to system security. This study showcases the susceptibility of deep learning models to adversarial attacks, which can achieve 100% attack success rate (refer to Table 5). The proposed method, EaTVul, encompasses six stages: identification of important samples using support vector machines, identification of important features using the attention mechanism, generation of adversarial data based on these features using ChatGPT, preparation of an adversarial attack pool, selection of seed data using a fuzzy genetic algorithm, and the execution of an evasion attack. Extensive experiments demonstrate the effectiveness of EaTVul, achieving an attack success rate of more than 83% when the snippet size is greater than 2. Furthermore, in most cases with a snippet size of 4, EaTVul achieves a 100% attack success rate. The findings of this research emphasize the necessity of robust defenses against adversarial attacks in software vulnerability detection.



## **36. Debiased Graph Poisoning Attack via Contrastive Surrogate Objective**

cs.LG

9 pages. Proceeding ACM International Conference on Information and  Knowledge Management (CIKM 2024) Proceeding

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19155v1) [paper-pdf](http://arxiv.org/pdf/2407.19155v1)

**Authors**: Kanghoon Yoon, Yeonjun In, Namkyeong Lee, Kibum Kim, Chanyoung Park

**Abstract**: Graph neural networks (GNN) are vulnerable to adversarial attacks, which aim to degrade the performance of GNNs through imperceptible changes on the graph. However, we find that in fact the prevalent meta-gradient-based attacks, which utilizes the gradient of the loss w.r.t the adjacency matrix, are biased towards training nodes. That is, their meta-gradient is determined by a training procedure of the surrogate model, which is solely trained on the training nodes. This bias manifests as an uneven perturbation, connecting two nodes when at least one of them is a labeled node, i.e., training node, while it is unlikely to connect two unlabeled nodes. However, these biased attack approaches are sub-optimal as they do not consider flipping edges between two unlabeled nodes at all. This means that they miss the potential attacked edges between unlabeled nodes that significantly alter the representation of a node. In this paper, we investigate the meta-gradients to uncover the root cause of the uneven perturbations of existing attacks. Based on our analysis, we propose a Meta-gradient-based attack method using contrastive surrogate objective (Metacon), which alleviates the bias in meta-gradient using a new surrogate loss. We conduct extensive experiments to show that Metacon outperforms existing meta gradient-based attack methods through benchmark datasets, while showing that alleviating the bias towards training nodes is effective in attacking the graph structure.



## **37. A Survey of Malware Detection Using Deep Learning**

cs.CR

**SubmitDate**: 2024-07-27    [abs](http://arxiv.org/abs/2407.19153v1) [paper-pdf](http://arxiv.org/pdf/2407.19153v1)

**Authors**: Ahmed Bensaoud, Jugal Kalita, Mahmoud Bensaoud

**Abstract**: The problem of malicious software (malware) detection and classification is a complex task, and there is no perfect approach. There is still a lot of work to be done. Unlike most other research areas, standard benchmarks are difficult to find for malware detection. This paper aims to investigate recent advances in malware detection on MacOS, Windows, iOS, Android, and Linux using deep learning (DL) by investigating DL in text and image classification, the use of pre-trained and multi-task learning models for malware detection approaches to obtain high accuracy and which the best approach if we have a standard benchmark dataset. We discuss the issues and the challenges in malware detection using DL classifiers by reviewing the effectiveness of these DL classifiers and their inability to explain their decisions and actions to DL developers presenting the need to use Explainable Machine Learning (XAI) or Interpretable Machine Learning (IML) programs. Additionally, we discuss the impact of adversarial attacks on deep learning models, negatively affecting their generalization capabilities and resulting in poor performance on unseen data. We believe there is a need to train and test the effectiveness and efficiency of the current state-of-the-art deep learning models on different malware datasets. We examine eight popular DL approaches on various datasets. This survey will help researchers develop a general understanding of malware recognition using deep learning.



## **38. Accurate and Scalable Detection and Investigation of Cyber Persistence Threats**

cs.CR

16 pages

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18832v1) [paper-pdf](http://arxiv.org/pdf/2407.18832v1)

**Authors**: Qi Liu, Muhammad Shoaib, Mati Ur Rehman, Kaibin Bao, Veit Hagenmeyer, Wajih Ul Hassan

**Abstract**: In Advanced Persistent Threat (APT) attacks, achieving stealthy persistence within target systems is often crucial for an attacker's success. This persistence allows adversaries to maintain prolonged access, often evading detection mechanisms. Recognizing its pivotal role in the APT lifecycle, this paper introduces Cyber Persistence Detector (CPD), a novel system dedicated to detecting cyber persistence through provenance analytics. CPD is founded on the insight that persistent operations typically manifest in two phases: the "persistence setup" and the subsequent "persistence execution". By causally relating these phases, we enhance our ability to detect persistent threats. First, CPD discerns setups signaling an impending persistent threat and then traces processes linked to remote connections to identify persistence execution activities. A key feature of our system is the introduction of pseudo-dependency edges (pseudo-edges), which effectively connect these disjoint phases using data provenance analysis, and expert-guided edges, which enable faster tracing and reduced log size. These edges empower us to detect persistence threats accurately and efficiently. Moreover, we propose a novel alert triage algorithm that further reduces false positives associated with persistence threats. Evaluations conducted on well-known datasets demonstrate that our system reduces the average false positive rate by 93% compared to state-of-the-art methods.



## **39. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

cs.DC

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2404.14250v4) [paper-pdf](http://arxiv.org/pdf/2404.14250v4)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.



## **40. Embodied Laser Attack:Leveraging Scene Priors to Achieve Agent-based Robust Non-contact Attacks**

cs.CV

9 pages, 7 figures, Accepted by ACM MM 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2312.09554v3) [paper-pdf](http://arxiv.org/pdf/2312.09554v3)

**Authors**: Yitong Sun, Yao Huang, Xingxing Wei

**Abstract**: As physical adversarial attacks become extensively applied in unearthing the potential risk of security-critical scenarios, especially in dynamic scenarios, their vulnerability to environmental variations has also been brought to light. The non-robust nature of physical adversarial attack methods brings less-than-stable performance consequently. Although methods such as EOT have enhanced the robustness of traditional contact attacks like adversarial patches, they fall short in practicality and concealment within dynamic environments such as traffic scenarios. Meanwhile, non-contact laser attacks, while offering enhanced adaptability, face constraints due to a limited optimization space for their attributes, rendering EOT less effective. This limitation underscores the necessity for developing a new strategy to augment the robustness of such practices. To address these issues, this paper introduces the Embodied Laser Attack (ELA), a novel framework that leverages the embodied intelligence paradigm of Perception-Decision-Control to dynamically tailor non-contact laser attacks. For the perception module, given the challenge of simulating the victim's view by full-image transformation, ELA has innovatively developed a local perspective transformation network, based on the intrinsic prior knowledge of traffic scenes and enables effective and efficient estimation. For the decision and control module, ELA trains an attack agent with data-driven reinforcement learning instead of adopting time-consuming heuristic algorithms, making it capable of instantaneously determining a valid attack strategy with the perceived information by well-designed rewards, which is then conducted by a controllable laser emitter. Experimentally, we apply our framework to diverse traffic scenarios both in the digital and physical world, verifying the effectiveness of our method under dynamic successive scenes.



## **41. Adversarial Robustification via Text-to-Image Diffusion Models**

cs.CV

Code is available at https://github.com/ChoiDae1/robustify-T2I

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18658v1) [paper-pdf](http://arxiv.org/pdf/2407.18658v1)

**Authors**: Daewon Choi, Jongheon Jeong, Huiwon Jang, Jinwoo Shin

**Abstract**: Adversarial robustness has been conventionally believed as a challenging property to encode for neural networks, requiring plenty of training data. In the recent paradigm of adopting off-the-shelf models, however, access to their training data is often infeasible or not practical, while most of such models are not originally trained concerning adversarial robustness. In this paper, we develop a scalable and model-agnostic solution to achieve adversarial robustness without using any data. Our intuition is to view recent text-to-image diffusion models as "adaptable" denoisers that can be optimized to specify target tasks. Based on this, we propose: (a) to initiate a denoise-and-classify pipeline that offers provable guarantees against adversarial attacks, and (b) to leverage a few synthetic reference images generated from the text-to-image model that enables novel adaptation schemes. Our experiments show that our data-free scheme applied to the pre-trained CLIP could improve the (provable) adversarial robustness of its diverse zero-shot classification derivatives (while maintaining their accuracy), significantly surpassing prior approaches that utilize the full training data. Not only for CLIP, we also demonstrate that our framework is easily applicable for robustifying other visual classifiers efficiently.



## **42. Robust VAEs via Generating Process of Noise Augmented Data**

cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18632v1) [paper-pdf](http://arxiv.org/pdf/2407.18632v1)

**Authors**: Hiroo Irobe, Wataru Aoki, Kimihiro Yamazaki, Yuhui Zhang, Takumi Nakagawa, Hiroki Waida, Yuichiro Wada, Takafumi Kanamori

**Abstract**: Advancing defensive mechanisms against adversarial attacks in generative models is a critical research topic in machine learning. Our study focuses on a specific type of generative models - Variational Auto-Encoders (VAEs). Contrary to common beliefs and existing literature which suggest that noise injection towards training data can make models more robust, our preliminary experiments revealed that naive usage of noise augmentation technique did not substantially improve VAE robustness. In fact, it even degraded the quality of learned representations, making VAEs more susceptible to adversarial perturbations. This paper introduces a novel framework that enhances robustness by regularizing the latent space divergence between original and noise-augmented data. Through incorporating a paired probabilistic prior into the standard variational lower bound, our method significantly boosts defense against adversarial attacks. Our empirical evaluations demonstrate that this approach, termed Robust Augmented Variational Auto-ENcoder (RAVEN), yields superior performance in resisting adversarial inputs on widely-recognized benchmark datasets.



## **43. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2305.17000v6) [paper-pdf](http://arxiv.org/pdf/2305.17000v6)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **44. Unveiling Privacy Vulnerabilities: Investigating the Role of Structure in Graph Data**

cs.LG

In KDD'24; with full appendix

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18564v1) [paper-pdf](http://arxiv.org/pdf/2407.18564v1)

**Authors**: Hanyang Yuan, Jiarong Xu, Cong Wang, Ziqi Yang, Chunping Wang, Keting Yin, Yang Yang

**Abstract**: The public sharing of user information opens the door for adversaries to infer private data, leading to privacy breaches and facilitating malicious activities. While numerous studies have concentrated on privacy leakage via public user attributes, the threats associated with the exposure of user relationships, particularly through network structure, are often neglected. This study aims to fill this critical gap by advancing the understanding and protection against privacy risks emanating from network structure, moving beyond direct connections with neighbors to include the broader implications of indirect network structural patterns. To achieve this, we first investigate the problem of Graph Privacy Leakage via Structure (GPS), and introduce a novel measure, the Generalized Homophily Ratio, to quantify the various mechanisms contributing to privacy breach risks in GPS. Based on this insight, we develop a novel graph private attribute inference attack, which acts as a pivotal tool for evaluating the potential for privacy leakage through network structures under worst-case scenarios. To protect users' private data from such vulnerabilities, we propose a graph data publishing method incorporating a learnable graph sampling technique, effectively transforming the original graph into a privacy-preserving version. Extensive experiments demonstrate that our attack model poses a significant threat to user privacy, and our graph data publishing method successfully achieves the optimal privacy-utility trade-off compared to baselines.



## **45. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

cs.CR

This work has been accepted by CCS 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2310.15469v3) [paper-pdf](http://arxiv.org/pdf/2310.15469v3)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.



## **46. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

cs.CV

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2406.18849v2) [paper-pdf](http://arxiv.org/pdf/2406.18849v2)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.



## **47. Machine Unlearning using a Multi-GAN based Model**

cs.LG

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18467v1) [paper-pdf](http://arxiv.org/pdf/2407.18467v1)

**Authors**: Amartya Hatua, Trung T. Nguyen, Andrew H. Sung

**Abstract**: This article presents a new machine unlearning approach that utilizes multiple Generative Adversarial Network (GAN) based models. The proposed method comprises two phases: i) data reorganization in which synthetic data using the GAN model is introduced with inverted class labels of the forget datasets, and ii) fine-tuning the pre-trained model. The GAN models consist of two pairs of generators and discriminators. The generator discriminator pairs generate synthetic data for the retain and forget datasets. Then, a pre-trained model is utilized to get the class labels of the synthetic datasets. The class labels of synthetic and original forget datasets are inverted. Finally, all combined datasets are used to fine-tune the pre-trained model to get the unlearned model. We have performed the experiments on the CIFAR-10 dataset and tested the unlearned models using Membership Inference Attacks (MIA). The inverted class labels procedure and synthetically generated data help to acquire valuable information that enables the model to outperform state-of-the-art models and other standard unlearning classifiers.



## **48. Disrupting Diffusion: Token-Level Attention Erasure Attack against Diffusion-based Customization**

cs.CV

Accepted by ACM MM2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2405.20584v2) [paper-pdf](http://arxiv.org/pdf/2405.20584v2)

**Authors**: Yisu Liu, Jinyang An, Wanqian Zhang, Dayan Wu, Jingzi Gu, Zheng Lin, Weiping Wang

**Abstract**: With the development of diffusion-based customization methods like DreamBooth, individuals now have access to train the models that can generate their personalized images. Despite the convenience, malicious users have misused these techniques to create fake images, thereby triggering a privacy security crisis. In light of this, proactive adversarial attacks are proposed to protect users against customization. The adversarial examples are trained to distort the customization model's outputs and thus block the misuse. In this paper, we propose DisDiff (Disrupting Diffusion), a novel adversarial attack method to disrupt the diffusion model outputs. We first delve into the intrinsic image-text relationships, well-known as cross-attention, and empirically find that the subject-identifier token plays an important role in guiding image generation. Thus, we propose the Cross-Attention Erasure module to explicitly "erase" the indicated attention maps and disrupt the text guidance. Besides,we analyze the influence of the sampling process of the diffusion model on Projected Gradient Descent (PGD) attack and introduce a novel Merit Sampling Scheduler to adaptively modulate the perturbation updating amplitude in a step-aware manner. Our DisDiff outperforms the state-of-the-art methods by 12.75% of FDFR scores and 7.25% of ISM scores across two facial benchmarks and two commonly used prompts on average.



## **49. Regret-Optimal Defense Against Stealthy Adversaries: A System Level Approach**

eess.SY

Accepted, IEEE Conference on Decision and Control (CDC), 2024

**SubmitDate**: 2024-07-26    [abs](http://arxiv.org/abs/2407.18448v1) [paper-pdf](http://arxiv.org/pdf/2407.18448v1)

**Authors**: Hiroyasu Tsukamoto, Joudi Hajar, Soon-Jo Chung, Fred Y. Hadaegh

**Abstract**: Modern control designs in robotics, aerospace, and cyber-physical systems heavily depend on real-world data obtained through the system outputs. In the face of system faults and malicious attacks, however, these outputs can be compromised to misrepresent some portion of the system information that critically affects their secure and trustworthy operation. In this paper, we introduce a novel regret-optimal control framework for designing controllers that render a linear system robust against stealthy attacks, including sensor and actuator attacks, as well as external disturbances. In particular, we establish (a) a convex optimization-based system metric to quantify the regret with the worst-case stealthy attack (the true performance minus the optimal performance in hindsight with the knowledge of the stealthy attack), which improves and adaptively interpolates $\mathcal{H}_2$ and $\mathcal{H}_{\infty}$ norms in the presence of stealthy adversaries, (b) an optimization problem for minimizing the regret of 1 expressed in the system level parameterization, which is useful for its localized and distributed implementation in large-scale systems, and (c) a rank-constrained optimization problem (i.e., optimization with a convex objective subject to convex constraints and rank constraints) equivalent to the optimization problem of (b). Finally, we conduct a numerical simulation which showcases the effectiveness of our approach.



## **50. Human-Interpretable Adversarial Prompt Attack on Large Language Models with Situational Context**

cs.CL

**SubmitDate**: 2024-07-25    [abs](http://arxiv.org/abs/2407.14644v2) [paper-pdf](http://arxiv.org/pdf/2407.14644v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous research on testing the vulnerabilities in Large Language Models (LLMs) using adversarial attacks has primarily focused on nonsensical prompt injections, which are easily detected upon manual or automated review (e.g., via byte entropy). However, the exploration of innocuous human-understandable malicious prompts augmented with adversarial injections remains limited. In this research, we explore converting a nonsensical suffix attack into a sensible prompt via a situation-driven contextual re-writing. This allows us to show suffix conversion without any gradients, using only LLMs to perform the attacks, and thus better understand the scope of possible risks. We combine an independent, meaningful adversarial insertion and situations derived from movies to check if this can trick an LLM. The situations are extracted from the IMDB dataset, and prompts are defined following a few-shot chain-of-thought prompting. Our approach demonstrates that a successful situation-driven attack can be executed on both open-source and proprietary LLMs. We find that across many LLMs, as few as 1 attempt produces an attack and that these attacks transfer between LLMs.



