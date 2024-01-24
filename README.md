# Latest Adversarial Attack Papers
**update at 2024-01-24 09:18:44**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Benchmarking the Robustness of Image Watermarks**

cs.CV

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.08573v2) [paper-pdf](http://arxiv.org/pdf/2401.08573v2)

**Authors**: Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, Furong Huang

**Abstract**: This paper investigates the weaknesses of image watermarking techniques. We present WAVES (Watermark Analysis Via Enhanced Stress-testing), a novel benchmark for assessing watermark robustness, overcoming the limitations of current evaluation methods.WAVES integrates detection and identification tasks, and establishes a standardized evaluation protocol comprised of a diverse range of stress tests. The attacks in WAVES range from traditional image distortions to advanced and novel variations of diffusive, and adversarial attacks. Our evaluation examines two pivotal dimensions: the degree of image quality degradation and the efficacy of watermark detection after attacks. We develop a series of Performance vs. Quality 2D plots, varying over several prominent image similarity metrics, which are then aggregated in a heuristically novel manner to paint an overall picture of watermark robustness and attack potency. Our comprehensive evaluation reveals previously undetected vulnerabilities of several modern watermarking algorithms. We envision WAVES as a toolkit for the future development of robust watermarking systems. The project is available at https://wavesbench.github.io/



## **2. NEUROSEC: FPGA-Based Neuromorphic Audio Security**

cs.CR

Audio processing, FPGA, Hardware Security, Neuromorphic Computing

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.12055v1) [paper-pdf](http://arxiv.org/pdf/2401.12055v1)

**Authors**: Murat Isik, Hiruna Vishwamith, Yusuf Sur, Kayode Inadagbo, I. Can Dikmen

**Abstract**: Neuromorphic systems, inspired by the complexity and functionality of the human brain, have gained interest in academic and industrial attention due to their unparalleled potential across a wide range of applications. While their capabilities herald innovation, it is imperative to underscore that these computational paradigms, analogous to their traditional counterparts, are not impervious to security threats. Although the exploration of neuromorphic methodologies for image and video processing has been rigorously pursued, the realm of neuromorphic audio processing remains in its early stages. Our results highlight the robustness and precision of our FPGA-based neuromorphic system. Specifically, our system showcases a commendable balance between desired signal and background noise, efficient spike rate encoding, and unparalleled resilience against adversarial attacks such as FGSM and PGD. A standout feature of our framework is its detection rate of 94%, which, when compared to other methodologies, underscores its greater capability in identifying and mitigating threats within 5.39 dB, a commendable SNR ratio. Furthermore, neuromorphic computing and hardware security serve many sensor domains in mission-critical and privacy-preserving applications.



## **3. The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images**

cs.CV

ICLR 2024. Code:  https://github.com/mazurowski-lab/intrinsic-properties

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.08865v2) [paper-pdf](http://arxiv.org/pdf/2401.08865v2)

**Authors**: Nicholas Konz, Maciej A. Mazurowski

**Abstract**: This paper investigates discrepancies in how neural networks learn from different imaging domains, which are commonly overlooked when adopting computer vision techniques from the domain of natural images to other specialized domains such as medical images. Recent works have found that the generalization error of a trained network typically increases with the intrinsic dimension ($d_{data}$) of its training set. Yet, the steepness of this relationship varies significantly between medical (radiological) and natural imaging domains, with no existing theoretical explanation. We address this gap in knowledge by establishing and empirically validating a generalization scaling law with respect to $d_{data}$, and propose that the substantial scaling discrepancy between the two considered domains may be at least partially attributed to the higher intrinsic "label sharpness" ($K_F$) of medical imaging datasets, a metric which we propose. Next, we demonstrate an additional benefit of measuring the label sharpness of a training set: it is negatively correlated with the trained model's adversarial robustness, which notably leads to models for medical images having a substantially higher vulnerability to adversarial attack. Finally, we extend our $d_{data}$ formalism to the related metric of learned representation intrinsic dimension ($d_{repr}$), derive a generalization scaling law with respect to $d_{repr}$, and show that $d_{data}$ serves as an upper bound for $d_{repr}$. Our theoretical results are supported by thorough experiments with six models and eleven natural and medical imaging datasets over a range of training set sizes. Our findings offer insights into the influence of intrinsic dataset properties on generalization, representation learning, and robustness in deep neural networks.



## **4. Diagnosis-guided Attack Recovery for Securing Robotic Vehicles from Sensor Deception Attacks**

cs.RO

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2209.04554v5) [paper-pdf](http://arxiv.org/pdf/2209.04554v5)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstract**: Sensors are crucial for perception and autonomous operation in robotic vehicles (RV). Unfortunately, RV sensors can be compromised by physical attacks such as sensor tampering or spoofing. In this paper, we present DeLorean, a unified framework for attack detection, attack diagnosis, and recovering RVs from sensor deception attacks (SDA). DeLorean can recover RVs even from strong SDAs in which the adversary targets multiple heterogeneous sensors simultaneously. We propose a novel attack diagnosis technique that inspects the attack-induced errors under SDAs, and identifies the targeted sensors using causal analysis. DeLorean then uses historic state information to selectively reconstruct physical states for compromised sensors, enabling targeted attack recovery under single or multi-sensor SDAs. We evaluate DeLorean on four real and two simulated RVs under SDAs targeting various sensors, and we find that it successfully recovers RVs from SDAs in 93% of the cases.



## **5. A Training-Free Defense Framework for Robust Learned Image Compression**

eess.IV

10 pages and 14 figures

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11902v1) [paper-pdf](http://arxiv.org/pdf/2401.11902v1)

**Authors**: Myungseo Song, Jinyoung Choi, Bohyung Han

**Abstract**: We study the robustness of learned image compression models against adversarial attacks and present a training-free defense technique based on simple image transform functions. Recent learned image compression models are vulnerable to adversarial attacks that result in poor compression rate, low reconstruction quality, or weird artifacts. To address the limitations, we propose a simple but effective two-way compression algorithm with random input transforms, which is conveniently applicable to existing image compression models. Unlike the na\"ive approaches, our approach preserves the original rate-distortion performance of the models on clean images. Moreover, the proposed algorithm requires no additional training or modification of existing models, making it more practical. We demonstrate the effectiveness of the proposed techniques through extensive experiments under multiple compression models, evaluation metrics, and attack scenarios.



## **6. Adversarial speech for voice privacy protection from Personalized Speech generation**

eess.AS

Accepted by icassp 2024

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11857v1) [paper-pdf](http://arxiv.org/pdf/2401.11857v1)

**Authors**: Shihao Chen, Liping Chen, Jie Zhang, KongAik Lee, Zhenhua Ling, Lirong Dai

**Abstract**: The rapid progress in personalized speech generation technology, including personalized text-to-speech (TTS) and voice conversion (VC), poses a challenge in distinguishing between generated and real speech for human listeners, resulting in an urgent demand in protecting speakers' voices from malicious misuse. In this regard, we propose a speaker protection method based on adversarial attacks. The proposed method perturbs speech signals by minimally altering the original speech while rendering downstream speech generation models unable to accurately generate the voice of the target speaker. For validation, we employ the open-source pre-trained YourTTS model for speech generation and protect the target speaker's speech in the white-box scenario. Automatic speaker verification (ASV) evaluations were carried out on the generated speech as the assessment of the voice protection capability. Our experimental results show that we successfully perturbed the speaker encoder of the YourTTS model using the gradient-based I-FGSM adversarial perturbation method. Furthermore, the adversarial perturbation is effective in preventing the YourTTS model from generating the speech of the target speaker. Audio samples can be found in https://voiceprivacy.github.io/Adeversarial-Speech-with-YourTTS.



## **7. Unraveling Attacks in Machine Learning-based IoT Ecosystems: A Survey and the Open Libraries Behind Them**

cs.CR

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11723v1) [paper-pdf](http://arxiv.org/pdf/2401.11723v1)

**Authors**: Chao Liu, Boxi Chen, Wei Shao, Chris Zhang, Kelvin Wong, Yi Zhang

**Abstract**: The advent of the Internet of Things (IoT) has brought forth an era of unprecedented connectivity, with an estimated 80 billion smart devices expected to be in operation by the end of 2025. These devices facilitate a multitude of smart applications, enhancing the quality of life and efficiency across various domains. Machine Learning (ML) serves as a crucial technology, not only for analyzing IoT-generated data but also for diverse applications within the IoT ecosystem. For instance, ML finds utility in IoT device recognition, anomaly detection, and even in uncovering malicious activities. This paper embarks on a comprehensive exploration of the security threats arising from ML's integration into various facets of IoT, spanning various attack types including membership inference, adversarial evasion, reconstruction, property inference, model extraction, and poisoning attacks. Unlike previous studies, our work offers a holistic perspective, categorizing threats based on criteria such as adversary models, attack targets, and key security attributes (confidentiality, availability, and integrity). We delve into the underlying techniques of ML attacks in IoT environment, providing a critical evaluation of their mechanisms and impacts. Furthermore, our research thoroughly assesses 65 libraries, both author-contributed and third-party, evaluating their role in safeguarding model and data privacy. We emphasize the availability and usability of these libraries, aiming to arm the community with the necessary tools to bolster their defenses against the evolving threat landscape. Through our comprehensive review and analysis, this paper seeks to contribute to the ongoing discourse on ML-based IoT security, offering valuable insights and practical solutions to secure ML models and data in the rapidly expanding field of artificial intelligence in IoT.



## **8. HashVFL: Defending Against Data Reconstruction Attacks in Vertical Federated Learning**

cs.CR

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2212.00325v2) [paper-pdf](http://arxiv.org/pdf/2212.00325v2)

**Authors**: Pengyu Qiu, Xuhong Zhang, Shouling Ji, Chong Fu, Xing Yang, Ting Wang

**Abstract**: Vertical Federated Learning (VFL) is a trending collaborative machine learning model training solution. Existing industrial frameworks employ secure multi-party computation techniques such as homomorphic encryption to ensure data security and privacy. Despite these efforts, studies have revealed that data leakage remains a risk in VFL due to the correlations between intermediate representations and raw data. Neural networks can accurately capture these correlations, allowing an adversary to reconstruct the data. This emphasizes the need for continued research into securing VFL systems.   Our work shows that hashing is a promising solution to counter data reconstruction attacks. The one-way nature of hashing makes it difficult for an adversary to recover data from hash codes. However, implementing hashing in VFL presents new challenges, including vanishing gradients and information loss. To address these issues, we propose HashVFL, which integrates hashing and simultaneously achieves learnability, bit balance, and consistency.   Experimental results indicate that HashVFL effectively maintains task performance while defending against data reconstruction attacks. It also brings additional benefits in reducing the degree of label leakage, mitigating adversarial attacks, and detecting abnormal inputs. We hope our work will inspire further research into the potential applications of HashVFL.



## **9. LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate**

cs.LG

AAAI 2024 main track. Code available on Github (see abstract).  Appendix is included in this updated version

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2312.13118v2) [paper-pdf](http://arxiv.org/pdf/2312.13118v2)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking \emph{given} pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose {\em Lipschitz Regularized Surrogate} (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.



## **10. Reducing Usefulness of Stolen Credentials in SSO Contexts**

cs.CR

8 pages, 5 figures

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11599v1) [paper-pdf](http://arxiv.org/pdf/2401.11599v1)

**Authors**: Sam Hays, Michael Sandborn, Dr. Jules White

**Abstract**: Approximately 61% of cyber attacks involve adversaries in possession of valid credentials. Attackers acquire credentials through various means, including phishing, dark web data drops, password reuse, etc. Multi-factor authentication (MFA) helps to thwart attacks that use valid credentials, but attackers still commonly breach systems by tricking users into accepting MFA step up requests through techniques, such as ``MFA Bombing'', where multiple requests are sent to a user until they accept one. Currently, there are several solutions to this problem, each with varying levels of security and increasing invasiveness on user devices. This paper proposes a token-based enrollment architecture that is less invasive to user devices than mobile device management, but still offers strong protection against use of stolen credentials and MFA attacks.



## **11. Thundernna: a white box adversarial attack**

cs.LG

10 pages, 5 figures

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2111.12305v2) [paper-pdf](http://arxiv.org/pdf/2111.12305v2)

**Authors**: Linfeng Ye, Shayan Mohajer Hamidi

**Abstract**: The existing work shows that the neural network trained by naive gradient-based optimization method is prone to adversarial attacks, adds small malicious on the ordinary input is enough to make the neural network wrong. At the same time, the attack against a neural network is the key to improving its robustness. The training against adversarial examples can make neural networks resist some kinds of adversarial attacks. At the same time, the adversarial attack against a neural network can also reveal some characteristics of the neural network, a complex high-dimensional non-linear function, as discussed in previous work.   In This project, we develop a first-order method to attack the neural network. Compare with other first-order attacks, our method has a much higher success rate. Furthermore, it is much faster than second-order attacks and multi-steps first-order attacks.



## **12. How Robust Are Energy-Based Models Trained With Equilibrium Propagation?**

cs.LG

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11543v1) [paper-pdf](http://arxiv.org/pdf/2401.11543v1)

**Authors**: Siddharth Mansingh, Michal Kucer, Garrett Kenyon, Juston Moore, Michael Teti

**Abstract**: Deep neural networks (DNNs) are easily fooled by adversarial perturbations that are imperceptible to humans. Adversarial training, a process where adversarial examples are added to the training set, is the current state-of-the-art defense against adversarial attacks, but it lowers the model's accuracy on clean inputs, is computationally expensive, and offers less robustness to natural noise. In contrast, energy-based models (EBMs), which were designed for efficient implementation in neuromorphic hardware and physical systems, incorporate feedback connections from each layer to the previous layer, yielding a recurrent, deep-attractor architecture which we hypothesize should make them naturally robust. Our work is the first to explore the robustness of EBMs to both natural corruptions and adversarial attacks, which we do using the CIFAR-10 and CIFAR-100 datasets. We demonstrate that EBMs are more robust than transformers and display comparable robustness to adversarially-trained DNNs on gradient-based (white-box) attacks, query-based (black-box) attacks, and natural perturbations without sacrificing clean accuracy, and without the need for adversarial training or additional training techniques.



## **13. Finding a Needle in the Adversarial Haystack: A Targeted Paraphrasing Approach For Uncovering Edge Cases with Minimal Distribution Distortion**

cs.CL

EACL 2024 - Main conference

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11373v1) [paper-pdf](http://arxiv.org/pdf/2401.11373v1)

**Authors**: Aly M. Kassem, Sherif Saad

**Abstract**: Adversarial attacks against NLP Deep Learning models are a significant concern. In particular, adversarial samples exploit the model's sensitivity to small input changes. While these changes appear insignificant on the semantics of the input sample, they result in significant decay in model performance. In this paper, we propose Targeted Paraphrasing via RL (TPRL), an approach to automatically learn a policy to generate challenging samples that most likely improve the model's performance. TPRL leverages FLAN T5, a language model, as a generator and employs a self learned policy using a proximal policy gradient to generate the adversarial examples automatically. TPRL's reward is based on the confusion induced in the classifier, preserving the original text meaning through a Mutual Implication score. We demonstrate and evaluate TPRL's effectiveness in discovering natural adversarial attacks and improving model performance through extensive experiments on four diverse NLP classification tasks via Automatic and Human evaluation. TPRL outperforms strong baselines, exhibits generalizability across classifiers and datasets, and combines the strengths of language modeling and reinforcement learning to generate diverse and influential adversarial examples.



## **14. Explainability-Driven Leaf Disease Classification Using Adversarial Training and Knowledge Distillation**

cs.CV

10 pages, 8 figures, Accepted by ICAART 2024

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.00334v3) [paper-pdf](http://arxiv.org/pdf/2401.00334v3)

**Authors**: Sebastian-Vasile Echim, Iulian-Marius Tăiatu, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: This work focuses on plant leaf disease classification and explores three crucial aspects: adversarial training, model explainability, and model compression. The models' robustness against adversarial attacks is enhanced through adversarial training, ensuring accurate classification even in the presence of threats. Leveraging explainability techniques, we gain insights into the model's decision-making process, improving trust and transparency. Additionally, we explore model compression techniques to optimize computational efficiency while maintaining classification performance. Through our experiments, we determine that on a benchmark dataset, the robustness can be the price of the classification accuracy with performance reductions of 3%-20% for regular tests and gains of 50%-70% for adversarial attack tests. We also demonstrate that a student model can be 15-25 times more computationally efficient for a slight performance reduction, distilling the knowledge of more complex models.



## **15. Robustness Against Adversarial Attacks via Learning Confined Adversarial Polytopes**

cs.LG

The paper has been accepted in ICASSP 2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.07991v2) [paper-pdf](http://arxiv.org/pdf/2401.07991v2)

**Authors**: Shayan Mohajer Hamidi, Linfeng Ye

**Abstract**: Deep neural networks (DNNs) could be deceived by generating human-imperceptible perturbations of clean samples. Therefore, enhancing the robustness of DNNs against adversarial attacks is a crucial task. In this paper, we aim to train robust DNNs by limiting the set of outputs reachable via a norm-bounded perturbation added to a clean sample. We refer to this set as adversarial polytope, and each clean sample has a respective adversarial polytope. Indeed, if the respective polytopes for all the samples are compact such that they do not intersect the decision boundaries of the DNN, then the DNN is robust against adversarial samples. Hence, the inner-working of our algorithm is based on learning \textbf{c}onfined \textbf{a}dversarial \textbf{p}olytopes (CAP). By conducting a thorough set of experiments, we demonstrate the effectiveness of CAP over existing adversarial robustness methods in improving the robustness of models against state-of-the-art attacks including AutoAttack.



## **16. Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts**

cs.CR

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2311.09127v2) [paper-pdf](http://arxiv.org/pdf/2311.09127v2)

**Authors**: Yuanwei Wu, Xiang Li, Yixin Liu, Pan Zhou, Lichao Sun

**Abstract**: Existing work on jailbreak Multimodal Large Language Models (MLLMs) has focused primarily on adversarial examples in model inputs, with less attention to vulnerabilities, especially in model API. To fill the research gap, we carry out the following work: 1) We discover a system prompt leakage vulnerability in GPT-4V. Through carefully designed dialogue, we successfully extract the internal system prompts of GPT-4V. This finding indicates potential exploitable security risks in MLLMs; 2) Based on the acquired system prompts, we propose a novel MLLM jailbreaking attack method termed SASP (Self-Adversarial Attack via System Prompt). By employing GPT-4 as a red teaming tool against itself, we aim to search for potential jailbreak prompts leveraging stolen system prompts. Furthermore, in pursuit of better performance, we also add human modification based on GPT-4's analysis, which further improves the attack success rate to 98.7\%; 3) We evaluated the effect of modifying system prompts to defend against jailbreaking attacks. Results show that appropriately designed system prompts can significantly reduce jailbreak success rates. Overall, our work provides new insights into enhancing MLLM security, demonstrating the important role of system prompts in jailbreaking. This finding could be leveraged to greatly facilitate jailbreak success rates while also holding the potential for defending against jailbreaks.



## **17. Susceptibility of Adversarial Attack on Medical Image Segmentation Models**

eess.IV

6 pages, 8 figures, presented at 2023 IEEE 20th International  Symposium on Biomedical Imaging (ISBI) conference

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11224v1) [paper-pdf](http://arxiv.org/pdf/2401.11224v1)

**Authors**: Zhongxuan Wang, Leo Xu

**Abstract**: The nature of deep neural networks has given rise to a variety of attacks, but little work has been done to address the effect of adversarial attacks on segmentation models trained on MRI datasets. In light of the grave consequences that such attacks could cause, we explore four models from the U-Net family and examine their responses to the Fast Gradient Sign Method (FGSM) attack. We conduct FGSM attacks on each of them and experiment with various schemes to conduct the attacks. In this paper, we find that medical imaging segmentation models are indeed vulnerable to adversarial attacks and that there is a negligible correlation between parameter size and adversarial attack success. Furthermore, we show that using a different loss function than the one used for training yields higher adversarial attack success, contrary to what the FGSM authors suggested. In future efforts, we will conduct the experiments detailed in this paper with more segmentation models and different attacks. We will also attempt to find ways to counteract the attacks by using model ensembles or special data augmentations. Our code is available at https://github.com/ZhongxuanWang/adv_attk



## **18. Generalizing Speaker Verification for Spoof Awareness in the Embedding Space**

cs.CR

To appear in IEEE/ACM Transactions on Audio, Speech, and Language  Processing

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11156v1) [paper-pdf](http://arxiv.org/pdf/2401.11156v1)

**Authors**: Xuechen Liu, Md Sahidullah, Kong Aik Lee, Tomi Kinnunen

**Abstract**: It is now well-known that automatic speaker verification (ASV) systems can be spoofed using various types of adversaries. The usual approach to counteract ASV systems against such attacks is to develop a separate spoofing countermeasure (CM) module to classify speech input either as a bonafide, or a spoofed utterance. Nevertheless, such a design requires additional computation and utilization efforts at the authentication stage. An alternative strategy involves a single monolithic ASV system designed to handle both zero-effort imposter (non-targets) and spoofing attacks. Such spoof-aware ASV systems have the potential to provide stronger protections and more economic computations. To this end, we propose to generalize the standalone ASV (G-SASV) against spoofing attacks, where we leverage limited training data from CM to enhance a simple backend in the embedding space, without the involvement of a separate CM module during the test (authentication) phase. We propose a novel yet simple backend classifier based on deep neural networks and conduct the study via domain adaptation and multi-task integration of spoof embeddings at the training stage. Experiments are conducted on the ASVspoof 2019 logical access dataset, where we improve the performance of statistical ASV backends on the joint (bonafide and spoofed) and spoofed conditions by a maximum of 36.2% and 49.8% in terms of equal error rates, respectively.



## **19. CARE: Ensemble Adversarial Robustness Evaluation Against Adaptive Attackers for Security Applications**

cs.CR

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11126v1) [paper-pdf](http://arxiv.org/pdf/2401.11126v1)

**Authors**: Hangsheng Zhang, Jiqiang Liu, Jinsong Dong

**Abstract**: Ensemble defenses, are widely employed in various security-related applications to enhance model performance and robustness. The widespread adoption of these techniques also raises many questions: Are general ensembles defenses guaranteed to be more robust than individuals? Will stronger adaptive attacks defeat existing ensemble defense strategies as the cybersecurity arms race progresses? Can ensemble defenses achieve adversarial robustness to different types of attacks simultaneously and resist the continually adjusted adaptive attacks? Unfortunately, these critical questions remain unresolved as there are no platforms for comprehensive evaluation of ensemble adversarial attacks and defenses in the cybersecurity domain. In this paper, we propose a general Cybersecurity Adversarial Robustness Evaluation (CARE) platform aiming to bridge this gap.



## **20. Universal Backdoor Attacks**

cs.LG

Accepted for publication at ICLR 2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2312.00157v2) [paper-pdf](http://arxiv.org/pdf/2312.00157v2)

**Authors**: Benjamin Schneider, Nils Lukas, Florian Kerschbaum

**Abstract**: Web-scraped datasets are vulnerable to data poisoning, which can be used for backdooring deep image classifiers during training. Since training on large datasets is expensive, a model is trained once and re-used many times. Unlike adversarial examples, backdoor attacks often target specific classes rather than any class learned by the model. One might expect that targeting many classes through a naive composition of attacks vastly increases the number of poison samples. We show this is not necessarily true and more efficient, universal data poisoning attacks exist that allow controlling misclassifications from any source class into any target class with a small increase in poison samples. Our idea is to generate triggers with salient characteristics that the model can learn. The triggers we craft exploit a phenomenon we call inter-class poison transferability, where learning a trigger from one class makes the model more vulnerable to learning triggers for other classes. We demonstrate the effectiveness and robustness of our universal backdoor attacks by controlling models with up to 6,000 classes while poisoning only 0.15% of the training dataset. Our source code is available at https://github.com/Ben-Schneider-code/Universal-Backdoor-Attacks.



## **21. Ensembler: Combating model inversion attacks using model ensemble during collaborative inference**

cs.CR

in submission

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10859v1) [paper-pdf](http://arxiv.org/pdf/2401.10859v1)

**Authors**: Dancheng Liu, Jinjun Xiong

**Abstract**: Deep learning models have exhibited remarkable performance across various domains. Nevertheless, the burgeoning model sizes compel edge devices to offload a significant portion of the inference process to the cloud. While this practice offers numerous advantages, it also raises critical concerns regarding user data privacy. In scenarios where the cloud server's trustworthiness is in question, the need for a practical and adaptable method to safeguard data privacy becomes imperative. In this paper, we introduce Ensembler, an extensible framework designed to substantially increase the difficulty of conducting model inversion attacks for adversarial parties. Ensembler leverages model ensembling on the adversarial server, running in parallel with existing approaches that introduce perturbations to sensitive data during colloborative inference. Our experiments demonstrate that when combined with even basic Gaussian noise, Ensembler can effectively shield images from reconstruction attacks, achieving recognition levels that fall below human performance in some strict settings, significantly outperforming baseline methods lacking the Ensembler framework.



## **22. Privacy-Preserving Neural Graph Databases**

cs.DB

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2312.15591v2) [paper-pdf](http://arxiv.org/pdf/2312.15591v2)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Yangqiu Song

**Abstract**: In the era of big data and rapidly evolving information systems, efficient and accurate data retrieval has become increasingly crucial. Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (graph DBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data. The usage of neural embedding storage and complex neural logical query answering provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the database. Malicious attackers can infer more sensitive information in the database using well-designed combinatorial queries, such as by comparing the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training due to the privacy concerns. In this work, inspired by the privacy protection in graph embeddings, we propose a privacy-preserving neural graph database (P-NGDB) to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to force the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries. Extensive experiment results on three datasets show that P-NGDB can effectively protect private information in the graph database while delivering high-quality public answers responses to queries.



## **23. Explainable and Transferable Adversarial Attack for ML-Based Network Intrusion Detectors**

cs.CR

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10691v1) [paper-pdf](http://arxiv.org/pdf/2401.10691v1)

**Authors**: Hangsheng Zhang, Dongqi Han, Yinlong Liu, Zhiliang Wang, Jiyan Sun, Shangyuan Zhuang, Jiqiang Liu, Jinsong Dong

**Abstract**: espite being widely used in network intrusion detection systems (NIDSs), machine learning (ML) has proven to be highly vulnerable to adversarial attacks. White-box and black-box adversarial attacks of NIDS have been explored in several studies. However, white-box attacks unrealistically assume that the attackers have full knowledge of the target NIDSs. Meanwhile, existing black-box attacks can not achieve high attack success rate due to the weak adversarial transferability between models (e.g., neural networks and tree models). Additionally, neither of them explains why adversarial examples exist and why they can transfer across models. To address these challenges, this paper introduces ETA, an Explainable Transfer-based Black-Box Adversarial Attack framework. ETA aims to achieve two primary objectives: 1) create transferable adversarial examples applicable to various ML models and 2) provide insights into the existence of adversarial examples and their transferability within NIDSs. Specifically, we first provide a general transfer-based adversarial attack method applicable across the entire ML space. Following that, we exploit a unique insight based on cooperative game theory and perturbation interpretations to explain adversarial examples and adversarial transferability. On this basis, we propose an Important-Sensitive Feature Selection (ISFS) method to guide the search for adversarial examples, achieving stronger transferability and ensuring traffic-space constraints.



## **24. FIMBA: Evaluating the Robustness of AI in Genomics via Feature Importance Adversarial Attacks**

cs.LG

15 pages, core code available at:  https://github.com/HeorhiiS/fimba-attack

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10657v1) [paper-pdf](http://arxiv.org/pdf/2401.10657v1)

**Authors**: Heorhii Skovorodnikov, Hoda Alkhzaimi

**Abstract**: With the steady rise of the use of AI in bio-technical applications and the widespread adoption of genomics sequencing, an increasing amount of AI-based algorithms and tools is entering the research and production stage affecting critical decision-making streams like drug discovery and clinical outcomes. This paper demonstrates the vulnerability of AI models often utilized downstream tasks on recognized public genomics datasets. We undermine model robustness by deploying an attack that focuses on input transformation while mimicking the real data and confusing the model decision-making, ultimately yielding a pronounced deterioration in model performance. Further, we enhance our approach by generating poisoned data using a variational autoencoder-based model. Our empirical findings unequivocally demonstrate a decline in model performance, underscored by diminished accuracy and an upswing in false positives and false negatives. Furthermore, we analyze the resulting adversarial samples via spectral analysis yielding conclusions for countermeasures against such attacks.



## **25. Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation**

cs.LG

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10590v1) [paper-pdf](http://arxiv.org/pdf/2401.10590v1)

**Authors**: Jialong Zhou, Xing Ai, Yuni Lai, Kai Zhou

**Abstract**: Signed graphs consist of edges and signs, which can be separated into structural information and balance-related information, respectively. Existing signed graph neural networks (SGNNs) typically rely on balance-related information to generate embeddings. Nevertheless, the emergence of recent adversarial attacks has had a detrimental impact on the balance-related information. Similar to how structure learning can restore unsigned graphs, balance learning can be applied to signed graphs by improving the balance degree of the poisoned graph. However, this approach encounters the challenge "Irreversibility of Balance-related Information" - while the balance degree improves, the restored edges may not be the ones originally affected by attacks, resulting in poor defense effectiveness. To address this challenge, we propose a robust SGNN framework called Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), which combines Graph Contrastive Learning principles with balance augmentation techniques. Experimental results demonstrate that BA-SGCL not only enhances robustness against existing adversarial attacks but also achieves superior performance on link sign prediction task across various datasets.



## **26. PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks**

cs.CR

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10586v1) [paper-pdf](http://arxiv.org/pdf/2401.10586v1)

**Authors**: Ping Guo, Zhiyuan Yang, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: Black-box query-based attacks constitute significant threats to Machine Learning as a Service (MLaaS) systems since they can generate adversarial examples without accessing the target model's architecture and parameters. Traditional defense mechanisms, such as adversarial training, gradient masking, and input transformations, either impose substantial computational costs or compromise the test accuracy of non-adversarial inputs. To address these challenges, we propose an efficient defense mechanism, PuriDefense, that employs random patch-wise purifications with an ensemble of lightweight purification models at a low level of inference cost. These models leverage the local implicit function and rebuild the natural image manifold. Our theoretical analysis suggests that this approach slows down the convergence of query-based attacks by incorporating randomness into purifications. Extensive experiments on CIFAR-10 and ImageNet validate the effectiveness of our proposed purifier-based defense mechanism, demonstrating significant improvements in robustness against query-based attacks.



## **27. Hijacking Attacks against Neural Networks by Analyzing Training Data**

cs.CR

Full version with major polishing, compared to the Usenix Security  2024 edition

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.09740v2) [paper-pdf](http://arxiv.org/pdf/2401.09740v2)

**Authors**: Yunjie Ge, Qian Wang, Huayang Huang, Qi Li, Cong Wang, Chao Shen, Lingchen Zhao, Peipei Jiang, Zheng Fang, Shenyi Zhang

**Abstract**: Backdoors and adversarial examples are the two primary threats currently faced by deep neural networks (DNNs). Both attacks attempt to hijack the model behaviors with unintended outputs by introducing (small) perturbations to the inputs. Backdoor attacks, despite the high success rates, often require a strong assumption, which is not always easy to achieve in reality. Adversarial example attacks, which put relatively weaker assumptions on attackers, often demand high computational resources, yet do not always yield satisfactory success rates when attacking mainstream black-box models in the real world. These limitations motivate the following research question: can model hijacking be achieved more simply, with a higher attack success rate and more reasonable assumptions? In this paper, we propose CleanSheet, a new model hijacking attack that obtains the high performance of backdoor attacks without requiring the adversary to tamper with the model training process. CleanSheet exploits vulnerabilities in DNNs stemming from the training data. Specifically, our key idea is to treat part of the clean training data of the target model as "poisoned data," and capture the characteristics of these data that are more sensitive to the model (typically called robust features) to construct "triggers." These triggers can be added to any input example to mislead the target model, similar to backdoor attacks. We validate the effectiveness of CleanSheet through extensive experiments on 5 datasets, 79 normally trained models, 68 pruned models, and 39 defensive models. Results show that CleanSheet exhibits performance comparable to state-of-the-art backdoor attacks, achieving an average attack success rate (ASR) of 97.5% on CIFAR-100 and 92.4% on GTSRB, respectively. Furthermore, CleanSheet consistently maintains a high ASR, when confronted with various mainstream backdoor defenses.



## **28. Adversarial Attack On Yolov5 For Traffic And Road Sign Detection**

cs.CV

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2306.06071v2) [paper-pdf](http://arxiv.org/pdf/2306.06071v2)

**Authors**: Sanyam Jain

**Abstract**: This paper implements and investigates popular adversarial attacks on the YOLOv5 Object Detection algorithm. The paper explores the vulnerability of the YOLOv5 to adversarial attacks in the context of traffic and road sign detection. The paper investigates the impact of different types of attacks, including the Limited memory Broyden Fletcher Goldfarb Shanno (L-BFGS), the Fast Gradient Sign Method (FGSM) attack, the Carlini and Wagner (C&W) attack, the Basic Iterative Method (BIM) attack, the Projected Gradient Descent (PGD) attack, One Pixel Attack, and the Universal Adversarial Perturbations attack on the accuracy of YOLOv5 in detecting traffic and road signs. The results show that YOLOv5 is susceptible to these attacks, with misclassification rates increasing as the magnitude of the perturbations increases. We also explain the results using saliency maps. The findings of this paper have important implications for the safety and reliability of object detection algorithms used in traffic and transportation systems, highlighting the need for more robust and secure models to ensure their effectiveness in real-world applications.



## **29. On the Adversarial Robustness of Camera-based 3D Object Detection**

cs.CV

Transactions on Machine Learning Research, 2024. ISSN 2835-8856

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2301.10766v2) [paper-pdf](http://arxiv.org/pdf/2301.10766v2)

**Authors**: Shaoyuan Xie, Zichao Li, Zeyu Wang, Cihang Xie

**Abstract**: In recent years, camera-based 3D object detection has gained widespread attention for its ability to achieve high performance with low computational cost. However, the robustness of these methods to adversarial attacks has not been thoroughly examined, especially when considering their deployment in safety-critical domains like autonomous driving. In this study, we conduct the first comprehensive investigation of the robustness of leading camera-based 3D object detection approaches under various adversarial conditions. We systematically analyze the resilience of these models under two attack settings: white-box and black-box; focusing on two primary objectives: classification and localization. Additionally, we delve into two types of adversarial attack techniques: pixel-based and patch-based. Our experiments yield four interesting findings: (a) bird's-eye-view-based representations exhibit stronger robustness against localization attacks; (b) depth-estimation-free approaches have the potential to show stronger robustness; (c) accurate depth estimation effectively improves robustness for depth-estimation-based methods; (d) incorporating multi-frame benign inputs can effectively mitigate adversarial attacks. We hope our findings can steer the development of future camera-based object detection models with enhanced adversarial robustness.



## **30. Differentially Private and Adversarially Robust Machine Learning: An Empirical Evaluation**

cs.LG

Accepted at PPAI-24: The 5th AAAI Workshop on Privacy-Preserving  Artificial Intelligence

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10405v1) [paper-pdf](http://arxiv.org/pdf/2401.10405v1)

**Authors**: Janvi Thakkar, Giulio Zizzo, Sergio Maffeis

**Abstract**: Malicious adversaries can attack machine learning models to infer sensitive information or damage the system by launching a series of evasion attacks. Although various work addresses privacy and security concerns, they focus on individual defenses, but in practice, models may undergo simultaneous attacks. This study explores the combination of adversarial training and differentially private training to defend against simultaneous attacks. While differentially-private adversarial training, as presented in DP-Adv, outperforms the other state-of-the-art methods in performance, it lacks formal privacy guarantees and empirical validation. Thus, in this work, we benchmark the performance of this technique using a membership inference attack and empirically show that the resulting approach is as private as non-robust private models. This work also highlights the need to explore privacy guarantees in dynamic training paradigms.



## **31. Vulnerabilities of Foundation Model Integrated Federated Learning Under Adversarial Threats**

cs.CR

Chen Wu and Xi Li are equal contribution. The corresponding author is  Jiaqi Wang

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10375v1) [paper-pdf](http://arxiv.org/pdf/2401.10375v1)

**Authors**: Chen Wu, Xi Li, Jiaqi Wang

**Abstract**: Federated Learning (FL) addresses critical issues in machine learning related to data privacy and security, yet suffering from data insufficiency and imbalance under certain circumstances. The emergence of foundation models (FMs) offers potential solutions to the limitations of existing FL frameworks, e.g., by generating synthetic data for model initialization. However, due to the inherent safety concerns of FMs, integrating FMs into FL could introduce new risks, which remains largely unexplored. To address this gap, we conduct the first investigation on the vulnerability of FM integrated FL (FM-FL) under adversarial threats. Based on a unified framework of FM-FL, we introduce a novel attack strategy that exploits safety issues of FM to compromise FL client models. Through extensive experiments with well-known models and benchmark datasets in both image and text domains, we reveal the high susceptibility of the FM-FL to this new threat under various FL configurations. Furthermore, we find that existing FL defense strategies offer limited protection against this novel attack approach. This research highlights the critical need for enhanced security measures in FL in the era of FMs.



## **32. Attack and Defense Analysis of Learned Image Compression**

eess.IV

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10345v1) [paper-pdf](http://arxiv.org/pdf/2401.10345v1)

**Authors**: Tianyu Zhu, Heming Sun, Xiankui Xiong, Xuanpeng Zhu, Yong Gong, Minge jing, Yibo Fan

**Abstract**: Learned image compression (LIC) is becoming more and more popular these years with its high efficiency and outstanding compression quality. Still, the practicality against modified inputs added with specific noise could not be ignored. White-box attacks such as FGSM and PGD use only gradient to compute adversarial images that mislead LIC models to output unexpected results. Our experiments compare the effects of different dimensions such as attack methods, models, qualities, and targets, concluding that in the worst case, there is a 61.55% decrease in PSNR or a 19.15 times increase in bit rate under the PGD attack. To improve their robustness, we conduct adversarial training by adding adversarial images into the training datasets, which obtains a 95.52% decrease in the R-D cost of the most vulnerable LIC model. We further test the robustness of H.266, whose better performance on reconstruction quality extends its possibility to defend one-step or iterative adversarial attacks.



## **33. Hacking Predictors Means Hacking Cars: Using Sensitivity Analysis to Identify Trajectory Prediction Vulnerabilities for Autonomous Driving Security**

cs.CR

10 pages, 6 figures, 1 tables

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10313v1) [paper-pdf](http://arxiv.org/pdf/2401.10313v1)

**Authors**: Marsalis Gibson, David Babazadeh, Claire Tomlin, Shankar Sastry

**Abstract**: Adversarial attacks on learning-based trajectory predictors have already been demonstrated. However, there are still open questions about the effects of perturbations on trajectory predictor inputs other than state histories, and how these attacks impact downstream planning and control. In this paper, we conduct a sensitivity analysis on two trajectory prediction models, Trajectron++ and AgentFormer. We observe that between all inputs, almost all of the perturbation sensitivities for Trajectron++ lie only within the most recent state history time point, while perturbation sensitivities for AgentFormer are spread across state histories over time. We additionally demonstrate that, despite dominant sensitivity on state history perturbations, an undetectable image map perturbation made with the Fast Gradient Sign Method can induce large prediction error increases in both models. Even though image maps may contribute slightly to the prediction output of both models, this result reveals that rather than being robust to adversarial image perturbations, trajectory predictors are susceptible to image attacks. Using an optimization-based planner and example perturbations crafted from sensitivity results, we show how this vulnerability can cause a vehicle to come to a sudden stop from moderate driving speeds.



## **34. Marrying Adapters and Mixup to Efficiently Enhance the Adversarial Robustness of Pre-Trained Language Models for Text Classification**

cs.CL

10 pages and 2 figures

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10111v1) [paper-pdf](http://arxiv.org/pdf/2401.10111v1)

**Authors**: Tuc Nguyen, Thai Le

**Abstract**: Existing works show that augmenting training data of neural networks using both clean and adversarial examples can enhance their generalizability under adversarial attacks. However, this training approach often leads to performance degradation on clean inputs. Additionally, it requires frequent re-training of the entire model to account for new attack types, resulting in significant and costly computations. Such limitations make adversarial training mechanisms less practical, particularly for complex Pre-trained Language Models (PLMs) with millions or even billions of parameters. To overcome these challenges while still harnessing the theoretical benefits of adversarial training, this study combines two concepts: (1) adapters, which enable parameter-efficient fine-tuning, and (2) Mixup, which train NNs via convex combinations of pairs data pairs. Intuitively, we propose to fine-tune PLMs through convex combinations of non-data pairs of fine-tuned adapters, one trained with clean and another trained with adversarial examples. Our experiments show that the proposed method achieves the best trade-off between training efficiency and predictive performance, both with and without attacks compared to other baselines on a variety of downstream tasks.



## **35. Power in Numbers: Robust reading comprehension by finetuning with four adversarial sentences per example**

cs.CL

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10091v1) [paper-pdf](http://arxiv.org/pdf/2401.10091v1)

**Authors**: Ariel Marcus

**Abstract**: Recent models have achieved human level performance on the Stanford Question Answering Dataset when using F1 scores to evaluate the reading comprehension task. Yet, teaching machines to comprehend text has not been solved in the general case. By appending one adversarial sentence to the context paragraph, past research has shown that the F1 scores from reading comprehension models drop almost in half. In this paper, I replicate past adversarial research with a new model, ELECTRA-Small, and demonstrate that the new model's F1 score drops from 83.9% to 29.2%. To improve ELECTRA-Small's resistance to this attack, I finetune the model on SQuAD v1.1 training examples with one to five adversarial sentences appended to the context paragraph. Like past research, I find that the finetuned model on one adversarial sentence does not generalize well across evaluation datasets. However, when finetuned on four or five adversarial sentences the model attains an F1 score of more than 70% on most evaluation datasets with multiple appended and prepended adversarial sentences. The results suggest that with enough examples we can make models robust to adversarial attacks.



## **36. HGAttack: Transferable Heterogeneous Graph Adversarial Attack**

cs.LG

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09945v1) [paper-pdf](http://arxiv.org/pdf/2401.09945v1)

**Authors**: He Zhao, Zhiwei Zeng, Yongwei Wang, Deheng Ye, Chunyan Miao

**Abstract**: Heterogeneous Graph Neural Networks (HGNNs) are increasingly recognized for their performance in areas like the web and e-commerce, where resilience against adversarial attacks is crucial. However, existing adversarial attack methods, which are primarily designed for homogeneous graphs, fall short when applied to HGNNs due to their limited ability to address the structural and semantic complexity of HGNNs. This paper introduces HGAttack, the first dedicated gray box evasion attack method for heterogeneous graphs. We design a novel surrogate model to closely resemble the behaviors of the target HGNN and utilize gradient-based methods for perturbation generation. Specifically, the proposed surrogate model effectively leverages heterogeneous information by extracting meta-path induced subgraphs and applying GNNs to learn node embeddings with distinct semantics from each subgraph. This approach improves the transferability of generated attacks on the target HGNN and significantly reduces memory costs. For perturbation generation, we introduce a semantics-aware mechanism that leverages subgraph gradient information to autonomously identify vulnerable edges across a wide range of relations within a constrained perturbation budget. We validate HGAttack's efficacy with comprehensive experiments on three datasets, providing empirical analyses of its generated perturbations. Outperforming baseline methods, HGAttack demonstrated significant efficacy in diminishing the performance of target HGNN models, affirming the effectiveness of our approach in evaluating the robustness of HGNNs against adversarial attacks.



## **37. Universally Robust Graph Neural Networks by Preserving Neighbor Similarity**

cs.LG

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09754v1) [paper-pdf](http://arxiv.org/pdf/2401.09754v1)

**Authors**: Yulin Zhu, Yuni Lai, Xing Ai, Kai Zhou

**Abstract**: Despite the tremendous success of graph neural networks in learning relational data, it has been widely investigated that graph neural networks are vulnerable to structural attacks on homophilic graphs. Motivated by this, a surge of robust models is crafted to enhance the adversarial robustness of graph neural networks on homophilic graphs. However, the vulnerability based on heterophilic graphs remains a mystery to us. To bridge this gap, in this paper, we start to explore the vulnerability of graph neural networks on heterophilic graphs and theoretically prove that the update of the negative classification loss is negatively correlated with the pairwise similarities based on the powered aggregated neighbor features. This theoretical proof explains the empirical observations that the graph attacker tends to connect dissimilar node pairs based on the similarities of neighbor features instead of ego features both on homophilic and heterophilic graphs. In this way, we novelly introduce a novel robust model termed NSPGNN which incorporates a dual-kNN graphs pipeline to supervise the neighbor similarity-guided propagation. This propagation utilizes the low-pass filter to smooth the features of node pairs along the positive kNN graphs and the high-pass filter to discriminate the features of node pairs along the negative kNN graphs. Extensive experiments on both homophilic and heterophilic graphs validate the universal robustness of NSPGNN compared to the state-of-the-art methods.



## **38. X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network**

cs.CR

This is the Accepted version of an article for publication in IEEE  TVT

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2303.12278v2) [paper-pdf](http://arxiv.org/pdf/2303.12278v2)

**Authors**: Seonghoon Jeong, Sangho Lee, Hwejae Lee, Huy Kang Kim

**Abstract**: Controller Area Network (CAN) is an essential networking protocol that connects multiple electronic control units (ECUs) in a vehicle. However, CAN-based in-vehicle networks (IVNs) face security risks owing to the CAN mechanisms. An adversary can sabotage a vehicle by leveraging the security risks if they can access the CAN bus. Thus, recent actions and cybersecurity regulations (e.g., UNR 155) require carmakers to implement intrusion detection systems (IDSs) in their vehicles. The IDS should detect cyberattacks and provide additional information to analyze conducted attacks. Although many IDSs have been proposed, considerations regarding their feasibility and explainability remain lacking. This study proposes X-CANIDS, which is a novel IDS for CAN-based IVNs. X-CANIDS dissects the payloads in CAN messages into human-understandable signals using a CAN database. The signals improve the intrusion detection performance compared with the use of bit representations of raw payloads. These signals also enable an understanding of which signal or ECU is under attack. X-CANIDS can detect zero-day attacks because it does not require any labeled dataset in the training phase. We confirmed the feasibility of the proposed method through a benchmark test on an automotive-grade embedded device with a GPU. The results of this work will be valuable to carmakers and researchers considering the installation of in-vehicle IDSs for their vehicles.



## **39. Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack**

cs.CV

9 pages, 5 figures

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09673v1) [paper-pdf](http://arxiv.org/pdf/2401.09673v1)

**Authors**: Zhongliang Guo, Kaixuan Wang, Weiye Li, Yifei Qian, Ognjen Arandjelović, Lei Fang

**Abstract**: Neural style transfer (NST) is widely adopted in computer vision to generate new images with arbitrary styles. This process leverages neural networks to merge aesthetic elements of a style image with the structural aspects of a content image into a harmoniously integrated visual result. However, unauthorized NST can exploit artwork. Such misuse raises socio-technical concerns regarding artists' rights and motivates the development of technical approaches for the proactive protection of original creations. Adversarial attack is a concept primarily explored in machine learning security. Our work introduces this technique to protect artists' intellectual property. In this paper Locally Adaptive Adversarial Color Attack (LAACA), a method for altering images in a manner imperceptible to the human eyes but disruptive to NST. Specifically, we design perturbations targeting image areas rich in high-frequency content, generated by disrupting intermediate features. Our experiments and user study confirm that by attacking NST using the proposed method results in visually worse neural style transfer, thus making it an effective solution for visual artwork protection.



## **40. MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks**

eess.IV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09624v1) [paper-pdf](http://arxiv.org/pdf/2401.09624v1)

**Authors**: Giovanni Pasqualino, Luca Guarnera, Alessandro Ortis, Sebastiano Battiato

**Abstract**: The progress in generative models, particularly Generative Adversarial Networks (GANs), opened new possibilities for image generation but raised concerns about potential malicious uses, especially in sensitive areas like medical imaging. This study introduces MITS-GAN, a novel approach to prevent tampering in medical images, with a specific focus on CT scans. The approach disrupts the output of the attacker's CT-GAN architecture by introducing imperceptible but yet precise perturbations. Specifically, the proposed approach involves the introduction of appropriate Gaussian noise to the input as a protective measure against various attacks. Our method aims to enhance tamper resistance, comparing favorably to existing techniques. Experimental results on a CT scan dataset demonstrate MITS-GAN's superior performance, emphasizing its ability to generate tamper-resistant images with negligible artifacts. As image tampering in medical domains poses life-threatening risks, our proactive approach contributes to the responsible and ethical use of generative models. This work provides a foundation for future research in countering cyber threats in medical imaging. Models and codes are publicly available at the following link \url{https://iplab.dmi.unict.it/MITS-GAN-2024/}.



## **41. Towards Scalable and Robust Model Versioning**

cs.LG

Accepted in IEEE SaTML 2024

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09574v1) [paper-pdf](http://arxiv.org/pdf/2401.09574v1)

**Authors**: Wenxin Ding, Arjun Nitin Bhagoji, Ben Y. Zhao, Haitao Zheng

**Abstract**: As the deployment of deep learning models continues to expand across industries, the threat of malicious incursions aimed at gaining access to these deployed models is on the rise. Should an attacker gain access to a deployed model, whether through server breaches, insider attacks, or model inversion techniques, they can then construct white-box adversarial attacks to manipulate the model's classification outcomes, thereby posing significant risks to organizations that rely on these models for critical tasks. Model owners need mechanisms to protect themselves against such losses without the necessity of acquiring fresh training data - a process that typically demands substantial investments in time and capital.   In this paper, we explore the feasibility of generating multiple versions of a model that possess different attack properties, without acquiring new training data or changing model architecture. The model owner can deploy one version at a time and replace a leaked version immediately with a new version. The newly deployed model version can resist adversarial attacks generated leveraging white-box access to one or all previously leaked versions. We show theoretically that this can be accomplished by incorporating parameterized hidden distributions into the model training data, forcing the model to learn task-irrelevant features uniquely defined by the chosen data. Additionally, optimal choices of hidden distributions can produce a sequence of model versions capable of resisting compound transferability attacks over time. Leveraging our analytical insights, we design and implement a practical model versioning method for DNN classifiers, which leads to significant robustness improvements over existing methods. We believe our work presents a promising direction for safeguarding DNN services beyond their initial deployment.



## **42. Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability**

cs.CV

Accepted as a conference paper in NeurIPS'2023. Code repo:  https://github.com/xavihart/Diff-PGD

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2305.16494v3) [paper-pdf](http://arxiv.org/pdf/2305.16494v3)

**Authors**: Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen

**Abstract**: Neural networks are known to be susceptible to adversarial samples: small variations of natural examples crafted to deliberately mislead the models. While they can be easily generated using gradient-based techniques in digital and physical scenarios, they often differ greatly from the actual data distribution of natural images, resulting in a trade-off between strength and stealthiness. In this paper, we propose a novel framework dubbed Diffusion-Based Projected Gradient Descent (Diff-PGD) for generating realistic adversarial samples. By exploiting a gradient guided by a diffusion model, Diff-PGD ensures that adversarial samples remain close to the original data distribution while maintaining their effectiveness. Moreover, our framework can be easily customized for specific tasks such as digital attacks, physical-world attacks, and style-based attacks. Compared with existing methods for generating natural-style adversarial samples, our framework enables the separation of optimizing adversarial loss from other surrogate losses (e.g., content/smoothness/style loss), making it more stable and controllable. Finally, we demonstrate that the samples generated using Diff-PGD have better transferability and anti-purification power than traditional gradient-based methods. Code will be released in https://github.com/xavihart/Diff-PGD



## **43. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

cs.CV

under review

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.06637v3) [paper-pdf](http://arxiv.org/pdf/2401.06637v3)

**Authors**: Peter Lorenz, Ricard Durall, Janis Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.



## **44. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

cs.CV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2312.04960v2) [paper-pdf](http://arxiv.org/pdf/2312.04960v2)

**Authors**: Xiaoyun Xu, Shujian Yu, Jingzheng Wu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) achieve superior performance on various tasks compared to convolutional neural networks (CNNs), but ViTs are also vulnerable to adversarial attacks. Adversarial training is one of the most successful methods to build robust CNN models. Thus, recent works explored new methodologies for adversarial training of ViTs based on the differences between ViTs and CNNs, such as better training strategies, preventing attention from focusing on a single block, or discarding low-attention embeddings. However, these methods still follow the design of traditional supervised adversarial training, limiting the potential of adversarial training on ViTs. This paper proposes a novel defense method, MIMIR, which aims to build a different adversarial training methodology by utilizing Masked Image Modeling at pre-training. We create an autoencoder that accepts adversarial examples as input but takes the clean examples as the modeling target. Then, we create a mutual information (MI) penalty following the idea of the Information Bottleneck. Among the two information source inputs and corresponding adversarial perturbation, the perturbation information is eliminated due to the constraint of the modeling target. Next, we provide a theoretical analysis of MIMIR using the bounds of the MI penalty. We also design two adaptive attacks when the adversary is aware of the MIMIR defense and show that MIMIR still performs well. The experimental results show that MIMIR improves (natural and adversarial) accuracy on average by 4.19% on CIFAR-10 and 5.52% on ImageNet-1K, compared to baselines. On Tiny-ImageNet, we obtained improved natural accuracy of 2.99\% on average and comparable adversarial accuracy. Our code and trained models are publicly available https://github.com/xiaoyunxxy/MIMIR.



## **45. Username Squatting on Online Social Networks: A Study on X**

cs.CR

Accepted at ACM ASIA Conference on Computer and Communications  Security (AsiaCCS), 2024

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09209v1) [paper-pdf](http://arxiv.org/pdf/2401.09209v1)

**Authors**: Anastasios Lepipas, Anastasia Borovykh, Soteris Demetriou

**Abstract**: Adversaries have been targeting unique identifiers to launch typo-squatting, mobile app squatting and even voice squatting attacks. Anecdotal evidence suggest that online social networks (OSNs) are also plagued with accounts that use similar usernames. This can be confusing to users but can also be exploited by adversaries. However, to date no study characterizes this problem on OSNs. In this work, we define the username squatting problem and design the first multi-faceted measurement study to characterize it on X. We develop a username generation tool (UsernameCrazy) to help us analyze hundreds of thousands of username variants derived from celebrity accounts. Our study reveals that thousands of squatted usernames have been suspended by X, while tens of thousands that still exist on the network are likely bots. Out of these, a large number share similar profile pictures and profile names to the original account signalling impersonation attempts. We found that squatted accounts are being mentioned by mistake in tweets hundreds of thousands of times and are even being prioritized in searches by the network's search recommendation algorithm exacerbating the negative impact squatted accounts can have in OSNs. We use our insights and take the first step to address this issue by designing a framework (SQUAD) that combines UsernameCrazy with a new classifier to efficiently detect suspicious squatted accounts. Our evaluation of SQUAD's prototype implementation shows that it can achieve 94% F1-score when trained on a small dataset.



## **46. Attack and Reset for Unlearning: Exploiting Adversarial Noise toward Machine Unlearning through Parameter Re-initialization**

cs.LG

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08998v1) [paper-pdf](http://arxiv.org/pdf/2401.08998v1)

**Authors**: Yoonhwa Jung, Ikhyun Cho, Shun-Hsiang Hsu, Julia Hockenmaier

**Abstract**: With growing concerns surrounding privacy and regulatory compliance, the concept of machine unlearning has gained prominence, aiming to selectively forget or erase specific learned information from a trained model. In response to this critical need, we introduce a novel approach called Attack-and-Reset for Unlearning (ARU). This algorithm leverages meticulously crafted adversarial noise to generate a parameter mask, effectively resetting certain parameters and rendering them unlearnable. ARU outperforms current state-of-the-art results on two facial machine-unlearning benchmark datasets, MUFAC and MUCAC. In particular, we present the steps involved in attacking and masking that strategically filter and re-initialize network parameters biased towards the forget set. Our work represents a significant advancement in rendering data unexploitable to deep learning models through parameter re-initialization, achieved by harnessing adversarial noise to craft a mask.



## **47. A GAN-based data poisoning framework against anomaly detection in vertical federated learning**

cs.LG

6 pages, 7 figures. This work has been submitted to the IEEE for  possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08984v1) [paper-pdf](http://arxiv.org/pdf/2401.08984v1)

**Authors**: Xiaolin Chen, Daoguang Zan, Wei Li, Bei Guan, Yongji Wang

**Abstract**: In vertical federated learning (VFL), commercial entities collaboratively train a model while preserving data privacy. However, a malicious participant's poisoning attack may degrade the performance of this collaborative model. The main challenge in achieving the poisoning attack is the absence of access to the server-side top model, leaving the malicious participant without a clear target model. To address this challenge, we introduce an innovative end-to-end poisoning framework P-GAN. Specifically, the malicious participant initially employs semi-supervised learning to train a surrogate target model. Subsequently, this participant employs a GAN-based method to produce adversarial perturbations to degrade the surrogate target model's performance. Finally, the generator is obtained and tailored for VFL poisoning. Besides, we develop an anomaly detection algorithm based on a deep auto-encoder (DAE), offering a robust defense mechanism to VFL scenarios. Through extensive experiments, we evaluate the efficacy of P-GAN and DAE, and further analyze the factors that influence their performance.



## **48. RandOhm: Mitigating Impedance Side-channel Attacks using Randomized Circuit Configurations**

cs.CR

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08925v1) [paper-pdf](http://arxiv.org/pdf/2401.08925v1)

**Authors**: Saleh Khalaj Monfared, Domenic Forte, Shahin Tajik

**Abstract**: Physical side-channel attacks can compromise the security of integrated circuits. Most of the physical side-channel attacks (e.g., power or electromagnetic) exploit the dynamic behavior of a chip, typically manifesting as changes in current consumption or voltage fluctuations where algorithmic countermeasures, such as masking, can effectively mitigate the attacks. However, as demonstrated recently, these mitigation techniques are not entirely effective against backscattered side-channel attacks such as impedance analysis. In the case of an impedance attack, an adversary exploits the data-dependent impedance variations of chip power delivery network (PDN) to extract secret information. In this work, we introduce RandOhm, which exploits moving target defense (MTD) strategy based on partial reconfiguration of mainstream FPGAs, to defend against impedance side-channel attacks. We demonstrate that the information leakage through the PDN impedance could be reduced via run-time reconfiguration of the secret-sensitive parts of the circuitry. Hence, by constantly randomizing the placement and routing of the circuit, one can decorrelate the data-dependent computation from the impedance value. To validate our claims, we present a systematic approach equipped with two different partial reconfiguration strategies on implementations of the AES cipher realized on 28-nm FPGAs. We investigate the overhead of our mitigation in terms of delay and performance and provide security analysis by performing non-profiled and profiled impedance analysis attacks against these implementations to demonstrate the resiliency of our approach.



## **49. PPR: Enhancing Dodging Attacks while Maintaining Impersonation Attacks on Face Recognition Systems**

cs.CV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08903v1) [paper-pdf](http://arxiv.org/pdf/2401.08903v1)

**Authors**: Fengfan Zhou, Heifei Ling

**Abstract**: Adversarial Attacks on Face Recognition (FR) encompass two types: impersonation attacks and evasion attacks. We observe that achieving a successful impersonation attack on FR does not necessarily ensure a successful dodging attack on FR in the black-box setting. Introducing a novel attack method named Pre-training Pruning Restoration Attack (PPR), we aim to enhance the performance of dodging attacks whilst avoiding the degradation of impersonation attacks. Our method employs adversarial example pruning, enabling a portion of adversarial perturbations to be set to zero, while tending to maintain the attack performance. By utilizing adversarial example pruning, we can prune the pre-trained adversarial examples and selectively free up certain adversarial perturbations. Thereafter, we embed adversarial perturbations in the pruned area, which enhances the dodging performance of the adversarial face examples. The effectiveness of our proposed attack method is demonstrated through our experimental results, showcasing its superior performance.



## **50. Whispering Pixels: Exploiting Uninitialized Register Accesses in Modern GPUs**

cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08881v1) [paper-pdf](http://arxiv.org/pdf/2401.08881v1)

**Authors**: Frederik Dermot Pustelnik, Xhani Marvin Saß, Jean-Pierre Seifert

**Abstract**: Graphic Processing Units (GPUs) have transcended their traditional use-case of rendering graphics and nowadays also serve as a powerful platform for accelerating ubiquitous, non-graphical rendering tasks. One prominent task is inference of neural networks, which process vast amounts of personal data, such as audio, text or images. Thus, GPUs became integral components for handling vast amounts of potentially confidential data, which has awakened the interest of security researchers. This lead to the discovery of various vulnerabilities in GPUs in recent years. In this paper, we uncover yet another vulnerability class in GPUs: We found that some GPU implementations lack proper register initialization routines before shader execution, leading to unintended register content leakage of previously executed shader kernels. We showcase the existence of the aforementioned vulnerability on products of 3 major vendors - Apple, NVIDIA and Qualcomm. The vulnerability poses unique challenges to an adversary due to opaque scheduling and register remapping algorithms present in the GPU firmware, complicating the reconstruction of leaked data. In order to illustrate the real-world impact of this flaw, we showcase how these challenges can be solved for attacking various workloads on the GPU. First, we showcase how uninitialized registers leak arbitrary pixel data processed by fragment shaders. We further implement information leakage attacks on intermediate data of Convolutional Neural Networks (CNNs) and present the attack's capability to leak and reconstruct the output of Large Language Models (LLMs).



