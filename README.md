# Latest Adversarial Attack Papers
**update at 2023-09-18 09:34:53**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. ICLEF: In-Context Learning with Expert Feedback for Explainable Style Transfer**

cs.CL

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2309.08583v1) [paper-pdf](http://arxiv.org/pdf/2309.08583v1)

**Authors**: Arkadiy Saakyan, Smaranda Muresan

**Abstract**: While state-of-the-art language models excel at the style transfer task, current work does not address explainability of style transfer systems. Explanations could be generated using large language models such as GPT-3.5 and GPT-4, but the use of such complex systems is inefficient when smaller, widely distributed, and transparent alternatives are available. We propose a framework to augment and improve a formality style transfer dataset with explanations via model distillation from ChatGPT. To further refine the generated explanations, we propose a novel way to incorporate scarce expert human feedback using in-context learning (ICLEF: In-Context Learning from Expert Feedback) by prompting ChatGPT to act as a critic to its own outputs. We use the resulting dataset of 9,960 explainable formality style transfer instances (e-GYAFC) to show that current openly distributed instruction-tuned models (and, in some settings, ChatGPT) perform poorly on the task, and that fine-tuning on our high-quality dataset leads to significant improvements as shown by automatic evaluation. In human evaluation, we show that models much smaller than ChatGPT fine-tuned on our data align better with expert preferences. Finally, we discuss two potential applications of models fine-tuned on the explainable style transfer task: interpretable authorship verification and interpretable adversarial attacks on AI-generated text detectors.



## **2. HINT: Healthy Influential-Noise based Training to Defend against Data Poisoning Attacks**

cs.LG

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2309.08549v1) [paper-pdf](http://arxiv.org/pdf/2309.08549v1)

**Authors**: Minh-Hao Van, Alycia N. Carey, Xintao Wu

**Abstract**: While numerous defense methods have been proposed to prohibit potential poisoning attacks from untrusted data sources, most research works only defend against specific attacks, which leaves many avenues for an adversary to exploit. In this work, we propose an efficient and robust training approach to defend against data poisoning attacks based on influence functions, named Healthy Influential-Noise based Training. Using influence functions, we craft healthy noise that helps to harden the classification model against poisoning attacks without significantly affecting the generalization ability on test data. In addition, our method can perform effectively when only a subset of the training data is modified, instead of the current method of adding noise to all examples that has been used in several previous works. We conduct comprehensive evaluations over two image datasets with state-of-the-art poisoning attacks under different realistic attack scenarios. Our empirical results show that HINT can efficiently protect deep learning models against the effect of both untargeted and targeted poisoning attacks.



## **3. Efficient and robust Sensor Placement in Complex Environments**

cs.LG

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2309.08545v1) [paper-pdf](http://arxiv.org/pdf/2309.08545v1)

**Authors**: Lukas Taus, Yen-Hsi Richard Tsai

**Abstract**: We address the problem of efficient and unobstructed surveillance or communication in complex environments. On one hand, one wishes to use a minimal number of sensors to cover the environment. On the other hand, it is often important to consider solutions that are robust against sensor failure or adversarial attacks. This paper addresses these challenges of designing minimal sensor sets that achieve multi-coverage constraints -- every point in the environment is covered by a prescribed number of sensors. We propose a greedy algorithm to achieve the objective. Further, we explore deep learning techniques to accelerate the evaluation of the objective function formulated in the greedy algorithm. The training of the neural network reveals that the geometric properties of the data significantly impact the network's performance, particularly at the end stage. By taking into account these properties, we discuss the differences in using greedy and $\epsilon$-greedy algorithms to generate data and their impact on the robustness of the network.



## **4. Federated Learning with Quantum Secure Aggregation**

quant-ph

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2207.07444v2) [paper-pdf](http://arxiv.org/pdf/2207.07444v2)

**Authors**: Yichi Zhang, Chao Zhang, Cai Zhang, Lixin Fan, Bei Zeng, Qiang Yang

**Abstract**: This article illustrates a novel Quantum Secure Aggregation (QSA) scheme that is designed to provide highly secure and efficient aggregation of local model parameters for federated learning. The scheme is secure in protecting private model parameters from being disclosed to semi-honest attackers by utilizing quantum bits i.e. qubits to represent model parameters. The proposed security mechanism ensures that any attempts to eavesdrop private model parameters can be immediately detected and stopped. The scheme is also efficient in terms of the low computational complexity of transmitting and aggregating model parameters through entangled qubits. Benefits of the proposed QSA scheme are showcased in a horizontal federated learning setting in which both a centralized and decentralized architectures are taken into account. It was empirically demonstrated that the proposed QSA can be readily applied to aggregate different types of local models including logistic regression (LR), convolutional neural networks (CNN) as well as quantum neural network (QNN), indicating the versatility of the QSA scheme. Performances of global models are improved to various extents with respect to local models obtained by individual participants, while no private model parameters are disclosed to semi-honest adversaries.



## **5. Diversifying the High-level Features for better Adversarial Transferability**

cs.CV

Accepted by BMVC 2023 (Oral)

**SubmitDate**: 2023-09-15    [abs](http://arxiv.org/abs/2304.10136v2) [paper-pdf](http://arxiv.org/pdf/2304.10136v2)

**Authors**: Zhiyuan Wang, Zeliang Zhang, Siyuan Liang, Xiaosen Wang

**Abstract**: Given the great threat of adversarial attacks against Deep Neural Networks (DNNs), numerous works have been proposed to boost transferability to attack real-world applications. However, existing attacks often utilize advanced gradient calculation or input transformation but ignore the white-box model. Inspired by the fact that DNNs are over-parameterized for superior performance, we propose diversifying the high-level features (DHF) for more transferable adversarial examples. In particular, DHF perturbs the high-level features by randomly transforming the high-level features and mixing them with the feature of benign samples when calculating the gradient at each iteration. Due to the redundancy of parameters, such transformation does not affect the classification performance but helps identify the invariant features across different models, leading to much better transferability. Empirical evaluations on ImageNet dataset show that DHF could effectively improve the transferability of existing momentum-based attacks. Incorporated into the input transformation-based attacks, DHF generates more transferable adversarial examples and outperforms the baselines with a clear margin when attacking several defense models, showing its generalization to various attacks and high effectiveness for boosting transferability. Code is available at https://github.com/Trustworthy-AI-Group/DHF.



## **6. Unleashing the Adversarial Facet of Software Debloating**

cs.CR

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.08058v1) [paper-pdf](http://arxiv.org/pdf/2309.08058v1)

**Authors**: Do-Men Su, Mohannad Alhanahnah

**Abstract**: Software debloating techniques are applied to craft a specialized version of the program based on the user's requirements and remove irrelevant code accordingly. The debloated programs presumably maintain better performance and reduce the attack surface in contrast to the original programs. This work unleashes the effectiveness of applying software debloating techniques on the robustness of machine learning systems in the malware classification domain. We empirically study how an adversarial can leverage software debloating techniques to mislead machine learning malware classification models. We apply software debloating techniques to generate adversarial examples and demonstrate these adversarial examples can reduce the detection rate of VirusTotal. Our study opens new directions for research into adversarial machine learning not only in malware detection/classification but also in other software domains.



## **7. CRYPTO-MINE: Cryptanalysis via Mutual Information Neural Estimation**

cs.CR

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.08019v1) [paper-pdf](http://arxiv.org/pdf/2309.08019v1)

**Authors**: Benjamin D. Kim, Vipindev Adat Vasudevan, Jongchan Woo, Alejandro Cohen, Rafael G. L. D'Oliveira, Thomas Stahlbuhk, Muriel Médard

**Abstract**: The use of Mutual Information (MI) as a measure to evaluate the efficiency of cryptosystems has an extensive history. However, estimating MI between unknown random variables in a high-dimensional space is challenging. Recent advances in machine learning have enabled progress in estimating MI using neural networks. This work presents a novel application of MI estimation in the field of cryptography. We propose applying this methodology directly to estimate the MI between plaintext and ciphertext in a chosen plaintext attack. The leaked information, if any, from the encryption could potentially be exploited by adversaries to compromise the computational security of the cryptosystem. We evaluate the efficiency of our approach by empirically analyzing multiple encryption schemes and baseline approaches. Furthermore, we extend the analysis to novel network coding-based cryptosystems that provide individual secrecy and study the relationship between information leakage and input distribution.



## **8. SLMIA-SR: Speaker-Level Membership Inference Attacks against Speaker Recognition Systems**

cs.CR

Accepted by the 31st Network and Distributed System Security (NDSS)  Symposium, 2024

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.07983v1) [paper-pdf](http://arxiv.org/pdf/2309.07983v1)

**Authors**: Guangke Chen, Yedi Zhang, Fu Song

**Abstract**: Membership inference attacks allow adversaries to determine whether a particular example was contained in the model's training dataset. While previous works have confirmed the feasibility of such attacks in various applications, none has focused on speaker recognition (SR), a promising voice-based biometric recognition technique. In this work, we propose SLMIA-SR, the first membership inference attack tailored to SR. In contrast to conventional example-level attack, our attack features speaker-level membership inference, i.e., determining if any voices of a given speaker, either the same as or different from the given inference voices, have been involved in the training of a model. It is particularly useful and practical since the training and inference voices are usually distinct, and it is also meaningful considering the open-set nature of SR, namely, the recognition speakers were often not present in the training data. We utilize intra-closeness and inter-farness, two training objectives of SR, to characterize the differences between training and non-training speakers and quantify them with two groups of features driven by carefully-established feature engineering to mount the attack. To improve the generalizability of our attack, we propose a novel mixing ratio training strategy to train attack models. To enhance the attack performance, we introduce voice chunk splitting to cope with the limited number of inference voices and propose to train attack models dependent on the number of inference voices. Our attack is versatile and can work in both white-box and black-box scenarios. Additionally, we propose two novel techniques to reduce the number of black-box queries while maintaining the attack performance. Extensive experiments demonstrate the effectiveness of SLMIA-SR.



## **9. Pareto Adversarial Robustness: Balancing Spatial Robustness and Sensitivity-based Robustness**

cs.LG

Published in SCIENCE CHINA Information Sciences (SCIS) 2023. Please  also refer to the published version in the Journal reference  https://www.sciengine.com/SCIS/doi/10.1007/s11432-022-3861-8

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2111.01996v2) [paper-pdf](http://arxiv.org/pdf/2111.01996v2)

**Authors**: Ke Sun, Mingjie Li, Zhouchen Lin

**Abstract**: Adversarial robustness, which primarily comprises sensitivity-based robustness and spatial robustness, plays an integral part in achieving robust generalization. In this paper, we endeavor to design strategies to achieve universal adversarial robustness. To achieve this, we first investigate the relatively less-explored realm of spatial robustness. Then, we integrate the existing spatial robustness methods by incorporating both local and global spatial vulnerability into a unified spatial attack and adversarial training approach. Furthermore, we present a comprehensive relationship between natural accuracy, sensitivity-based robustness, and spatial robustness, supported by strong evidence from the perspective of robust representation. Crucially, to reconcile the interplay between the mutual impacts of various robustness components into one unified framework, we incorporate the \textit{Pareto criterion} into the adversarial robustness analysis, yielding a novel strategy called Pareto Adversarial Training for achieving universal robustness. The resulting Pareto front, which delineates the set of optimal solutions, provides an optimal balance between natural accuracy and various adversarial robustness. This sheds light on solutions for achieving universal robustness in the future. To the best of our knowledge, we are the first to consider universal adversarial robustness via multi-objective optimization.



## **10. What Matters to Enhance Traffic Rule Compliance of Imitation Learning for Automated Driving**

cs.CV

8 pages, 2 figures

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.07808v1) [paper-pdf](http://arxiv.org/pdf/2309.07808v1)

**Authors**: Hongkuan Zhou, Aifen Sui, Wei Cao, Letian Shi

**Abstract**: More research attention has recently been given to end-to-end autonomous driving technologies where the entire driving pipeline is replaced with a single neural network because of its simpler structure and faster inference time. Despite this appealing approach largely reducing the components in driving pipeline, its simplicity also leads to interpretability problems and safety issues arXiv:2003.06404. The trained policy is not always compliant with the traffic rules and it is also hard to discover the reason for the misbehavior because of the lack of intermediate outputs. Meanwhile, Sensors are also critical to autonomous driving's security and feasibility to perceive the surrounding environment under complex driving scenarios. In this paper, we proposed P-CSG, a novel penalty-based imitation learning approach with cross semantics generation sensor fusion technologies to increase the overall performance of End-to-End Autonomous Driving. We conducted an assessment of our model's performance using the Town 05 Long benchmark, achieving an impressive driving score improvement of over 15%. Furthermore, we conducted robustness evaluations against adversarial attacks like FGSM and Dot attacks, revealing a substantial increase in robustness compared to baseline models.More detailed information, such as code-based resources, ablation studies and videos can be found at https://hk-zh.github.io/p-csg-plus.



## **11. TrojViT: Trojan Insertion in Vision Transformers**

cs.LG

10 pages, 4 figures, 11 tables

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2208.13049v4) [paper-pdf](http://arxiv.org/pdf/2208.13049v4)

**Authors**: Mengxin Zheng, Qian Lou, Lei Jiang

**Abstract**: Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform backdoor attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Na\"ively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack $TrojViT$. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses minimum-tuned parameter update to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify $99.64\%$ of test images to a target class by flipping $345$ bits on a ViT for ImageNet.Our codes are available at https://github.com/mxzheng/TrojViT



## **12. Physical Invisible Backdoor Based on Camera Imaging**

cs.CV

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.07428v1) [paper-pdf](http://arxiv.org/pdf/2309.07428v1)

**Authors**: Yusheng Guo, Nan Zhong, Zhenxing Qian, Xinpeng Zhang

**Abstract**: Backdoor attack aims to compromise a model, which returns an adversary-wanted output when a specific trigger pattern appears yet behaves normally for clean inputs. Current backdoor attacks require changing pixels of clean images, which results in poor stealthiness of attacks and increases the difficulty of the physical implementation. This paper proposes a novel physical invisible backdoor based on camera imaging without changing nature image pixels. Specifically, a compromised model returns a target label for images taken by a particular camera, while it returns correct results for other images. To implement and evaluate the proposed backdoor, we take shots of different objects from multi-angles using multiple smartphones to build a new dataset of 21,500 images. Conventional backdoor attacks work ineffectively with some classical models, such as ResNet18, over the above-mentioned dataset. Therefore, we propose a three-step training strategy to mount the backdoor attack. First, we design and train a camera identification model with the phone IDs to extract the camera fingerprint feature. Subsequently, we elaborate a special network architecture, which is easily compromised by our backdoor attack, by leveraging the attributes of the CFA interpolation algorithm and combining it with the feature extraction block in the camera identification model. Finally, we transfer the backdoor from the elaborated special network architecture to the classical architecture model via teacher-student distillation learning. Since the trigger of our method is related to the specific phone, our attack works effectively in the physical world. Experiment results demonstrate the feasibility of our proposed approach and robustness against various backdoor defenses.



## **13. Client-side Gradient Inversion Against Federated Learning from Poisoning**

cs.CR

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.07415v1) [paper-pdf](http://arxiv.org/pdf/2309.07415v1)

**Authors**: Jiaheng Wei, Yanjun Zhang, Leo Yu Zhang, Chao Chen, Shirui Pan, Kok-Leong Ong, Jun Zhang, Yang Xiang

**Abstract**: Federated Learning (FL) enables distributed participants (e.g., mobile devices) to train a global model without sharing data directly to a central server. Recent studies have revealed that FL is vulnerable to gradient inversion attack (GIA), which aims to reconstruct the original training samples and poses high risk against the privacy of clients in FL. However, most existing GIAs necessitate control over the server and rely on strong prior knowledge including batch normalization and data distribution information. In this work, we propose Client-side poisoning Gradient Inversion (CGI), which is a novel attack method that can be launched from clients. For the first time, we show the feasibility of a client-side adversary with limited knowledge being able to recover the training samples from the aggregated global model. We take a distinct approach in which the adversary utilizes a malicious model that amplifies the loss of a specific targeted class of interest. When honest clients employ the poisoned global model, the gradients of samples belonging to the targeted class are magnified, making them the dominant factor in the aggregated update. This enables the adversary to effectively reconstruct the private input belonging to other clients using the aggregated update. In addition, our CGI also features its ability to remain stealthy against Byzantine-robust aggregation rules (AGRs). By optimizing malicious updates and blending benign updates with a malicious replacement vector, our method remains undetected by these defense mechanisms. To evaluate the performance of CGI, we conduct experiments on various benchmark datasets, considering representative Byzantine-robust AGRs, and exploring diverse FL settings with different levels of adversary knowledge about the data. Our results demonstrate that CGI consistently and successfully extracts training input in all tested scenarios.



## **14. COVER: A Heuristic Greedy Adversarial Attack on Prompt-based Learning in Language Models**

cs.CL

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2306.05659v3) [paper-pdf](http://arxiv.org/pdf/2306.05659v3)

**Authors**: Zihao Tan, Qingliang Chen, Wenbin Zhu, Yongjian Huang

**Abstract**: Prompt-based learning has been proved to be an effective way in pre-trained language models (PLMs), especially in low-resource scenarios like few-shot settings. However, the trustworthiness of PLMs is of paramount significance and potential vulnerabilities have been shown in prompt-based templates that could mislead the predictions of language models, causing serious security concerns. In this paper, we will shed light on some vulnerabilities of PLMs, by proposing a prompt-based adversarial attack on manual templates in black box scenarios. First of all, we design character-level and word-level heuristic approaches to break manual templates separately. Then we present a greedy algorithm for the attack based on the above heuristic destructive approaches. Finally, we evaluate our approach with the classification tasks on three variants of BERT series models and eight datasets. And comprehensive experimental results justify the effectiveness of our approach in terms of attack success rate and attack speed.



## **15. Semantic Adversarial Attacks via Diffusion Models**

cs.CV

To appear in BMVC 2023

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.07398v1) [paper-pdf](http://arxiv.org/pdf/2309.07398v1)

**Authors**: Chenan Wang, Jinhao Duan, Chaowei Xiao, Edward Kim, Matthew Stamm, Kaidi Xu

**Abstract**: Traditional adversarial attacks concentrate on manipulating clean examples in the pixel space by adding adversarial perturbations. By contrast, semantic adversarial attacks focus on changing semantic attributes of clean examples, such as color, context, and features, which are more feasible in the real world. In this paper, we propose a framework to quickly generate a semantic adversarial attack by leveraging recent diffusion models since semantic information is included in the latent space of well-trained diffusion models. Then there are two variants of this framework: 1) the Semantic Transformation (ST) approach fine-tunes the latent space of the generated image and/or the diffusion model itself; 2) the Latent Masking (LM) approach masks the latent space with another target image and local backpropagation-based interpretation methods. Additionally, the ST approach can be applied in either white-box or black-box settings. Extensive experiments are conducted on CelebA-HQ and AFHQ datasets, and our framework demonstrates great fidelity, generalizability, and transferability compared to other baselines. Our approaches achieve approximately 100% attack success rate in multiple settings with the best FID as 36.61. Code is available at https://github.com/steven202/semantic_adv_via_dm.



## **16. Deep Nonparametric Convexified Filtering for Computational Photography, Image Synthesis and Adversarial Defense**

cs.CV

**SubmitDate**: 2023-09-14    [abs](http://arxiv.org/abs/2309.06724v2) [paper-pdf](http://arxiv.org/pdf/2309.06724v2)

**Authors**: Jianqiao Wangni

**Abstract**: We aim to provide a general framework of for computational photography that recovers the real scene from imperfect images, via the Deep Nonparametric Convexified Filtering (DNCF). It is consists of a nonparametric deep network to resemble the physical equations behind the image formation, such as denoising, super-resolution, inpainting, and flash. DNCF has no parameterization dependent on training data, therefore has a strong generalization and robustness to adversarial image manipulation. During inference, we also encourage the network parameters to be nonnegative and create a bi-convex function on the input and parameters, and this adapts to second-order optimization algorithms with insufficient running time, having 10X acceleration over Deep Image Prior. With these tools, we empirically verify its capability to defend image classification deep networks against adversary attack algorithms in real-time.



## **17. BAARD: Blocking Adversarial Examples by Testing for Applicability, Reliability and Decidability**

cs.LG

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2105.00495v2) [paper-pdf](http://arxiv.org/pdf/2105.00495v2)

**Authors**: Xinglong Chang, Katharina Dost, Kaiqi Zhao, Ambra Demontis, Fabio Roli, Gill Dobbie, Jörg Wicker

**Abstract**: Adversarial defenses protect machine learning models from adversarial attacks, but are often tailored to one type of model or attack. The lack of information on unknown potential attacks makes detecting adversarial examples challenging. Additionally, attackers do not need to follow the rules made by the defender. To address this problem, we take inspiration from the concept of Applicability Domain in cheminformatics. Cheminformatics models struggle to make accurate predictions because only a limited number of compounds are known and available for training. Applicability Domain defines a domain based on the known compounds and rejects any unknown compound that falls outside the domain. Similarly, adversarial examples start as harmless inputs, but can be manipulated to evade reliable classification by moving outside the domain of the classifier. We are the first to identify the similarity between Applicability Domain and adversarial detection. Instead of focusing on unknown attacks, we focus on what is known, the training data. We propose a simple yet robust triple-stage data-driven framework that checks the input globally and locally, and confirms that they are coherent with the model's output. This framework can be applied to any classification model and is not limited to specific attacks. We demonstrate these three stages work as one unit, effectively detecting various attacks, even for a white-box scenario.



## **18. RAIN: Your Language Models Can Align Themselves without Finetuning**

cs.CL

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.07124v1) [paper-pdf](http://arxiv.org/pdf/2309.07124v1)

**Authors**: Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, Hongyang Zhang

**Abstract**: Large language models (LLMs) often demonstrate inconsistencies with human preferences. Previous research gathered human preference data and then aligned the pre-trained models using reinforcement learning or instruction tuning, the so-called finetuning step. In contrast, aligning frozen LLMs without any extra data is more appealing. This work explores the potential of the latter setting. We discover that by integrating self-evaluation and rewind mechanisms, unaligned LLMs can directly produce responses consistent with human preferences via self-boosting. We introduce a novel inference method, Rewindable Auto-regressive INference (RAIN), that allows pre-trained LLMs to evaluate their own generation and use the evaluation results to guide backward rewind and forward generation for AI safety. Notably, RAIN operates without the need of extra data for model alignment and abstains from any training, gradient computation, or parameter updates; during the self-evaluation phase, the model receives guidance on which human preference to align with through a fixed-template prompt, eliminating the need to modify the initial prompt. Experimental results evaluated by GPT-4 and humans demonstrate the effectiveness of RAIN: on the HH dataset, RAIN improves the harmlessness rate of LLaMA 30B over vanilla inference from 82% to 97%, while maintaining the helpfulness rate. Under the leading adversarial attack llm-attacks on Vicuna 33B, RAIN establishes a new defense baseline by reducing the attack success rate from 94% to 19%.



## **19. Hardening RGB-D Object Recognition Systems against Adversarial Patch Attacks**

cs.CV

Accepted for publication in the Information Sciences journal

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.07106v1) [paper-pdf](http://arxiv.org/pdf/2309.07106v1)

**Authors**: Yang Zheng, Luca Demetrio, Antonio Emanuele Cinà, Xiaoyi Feng, Zhaoqiang Xia, Xiaoyue Jiang, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: RGB-D object recognition systems improve their predictive performances by fusing color and depth information, outperforming neural network architectures that rely solely on colors. While RGB-D systems are expected to be more robust to adversarial examples than RGB-only systems, they have also been proven to be highly vulnerable. Their robustness is similar even when the adversarial examples are generated by altering only the original images' colors. Different works highlighted the vulnerability of RGB-D systems; however, there is a lacking of technical explanations for this weakness. Hence, in our work, we bridge this gap by investigating the learned deep representation of RGB-D systems, discovering that color features make the function learned by the network more complex and, thus, more sensitive to small perturbations. To mitigate this problem, we propose a defense based on a detection mechanism that makes RGB-D systems more robust against adversarial examples. We empirically show that this defense improves the performances of RGB-D systems against adversarial examples even when they are computed ad-hoc to circumvent this detection mechanism, and that is also more effective than adversarial training.



## **20. Mitigating Adversarial Attacks in Federated Learning with Trusted Execution Environments**

cs.LG

12 pages, 4 figures, to be published in Proceedings 23rd  International Conference on Distributed Computing Systems. arXiv admin note:  substantial text overlap with arXiv:2308.04373

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.07197v1) [paper-pdf](http://arxiv.org/pdf/2309.07197v1)

**Authors**: Simon Queyrut, Valerio Schiavoni, Pascal Felber

**Abstract**: The main premise of federated learning (FL) is that machine learning model updates are computed locally to preserve user data privacy. This approach avoids by design user data to ever leave the perimeter of their device. Once the updates aggregated, the model is broadcast to all nodes in the federation. However, without proper defenses, compromised nodes can probe the model inside their local memory in search for adversarial examples, which can lead to dangerous real-world scenarios. For instance, in image-based applications, adversarial examples consist of images slightly perturbed to the human eye getting misclassified by the local model. These adversarial images are then later presented to a victim node's counterpart model to replay the attack. Typical examples harness dissemination strategies such as altered traffic signs (patch attacks) no longer recognized by autonomous vehicles or seemingly unaltered samples that poison the local dataset of the FL scheme to undermine its robustness. Pelta is a novel shielding mechanism leveraging Trusted Execution Environments (TEEs) that reduce the ability of attackers to craft adversarial samples. Pelta masks inside the TEE the first part of the back-propagation chain rule, typically exploited by attackers to craft the malicious samples. We evaluate Pelta on state-of-the-art accurate models using three well-established datasets: CIFAR-10, CIFAR-100 and ImageNet. We show the effectiveness of Pelta in mitigating six white-box state-of-the-art adversarial attacks, such as Projected Gradient Descent, Momentum Iterative Method, Auto Projected Gradient Descent, the Carlini & Wagner attack. In particular, Pelta constitutes the first attempt at defending an ensemble model against the Self-Attention Gradient attack to the best of our knowledge. Our code is available to the research community at https://github.com/queyrusi/Pelta.



## **21. Differentiable JPEG: The Devil is in the Details**

cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.06978v1) [paper-pdf](http://arxiv.org/pdf/2309.06978v1)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.



## **22. PhantomSound: Black-Box, Query-Efficient Audio Adversarial Attack via Split-Second Phoneme Injection**

cs.CR

RAID 2023

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.06960v1) [paper-pdf](http://arxiv.org/pdf/2309.06960v1)

**Authors**: Hanqing Guo, Guangjing Wang, Yuanda Wang, Bocheng Chen, Qiben Yan, Li Xiao

**Abstract**: In this paper, we propose PhantomSound, a query-efficient black-box attack toward voice assistants. Existing black-box adversarial attacks on voice assistants either apply substitution models or leverage the intermediate model output to estimate the gradients for crafting adversarial audio samples. However, these attack approaches require a significant amount of queries with a lengthy training stage. PhantomSound leverages the decision-based attack to produce effective adversarial audios, and reduces the number of queries by optimizing the gradient estimation. In the experiments, we perform our attack against 4 different speech-to-text APIs under 3 real-world scenarios to demonstrate the real-time attack impact. The results show that PhantomSound is practical and robust in attacking 5 popular commercial voice controllable devices over the air, and is able to bypass 3 liveness detection mechanisms with >95% success rate. The benchmark result shows that PhantomSound can generate adversarial examples and launch the attack in a few minutes. We significantly enhance the query efficiency and reduce the cost of a successful untargeted and targeted adversarial attack by 93.1% and 65.5% compared with the state-of-the-art black-box attacks, using merely ~300 queries (~5 minutes) and ~1,500 queries (~25 minutes), respectively.



## **23. Improving Visual Quality and Transferability of Adversarial Attacks on Face Recognition Simultaneously with Adversarial Restoration**

cs.CV

\copyright 2023 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2309.01582v3) [paper-pdf](http://arxiv.org/pdf/2309.01582v3)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Ping Li

**Abstract**: Adversarial face examples possess two critical properties: Visual Quality and Transferability. However, existing approaches rarely address these properties simultaneously, leading to subpar results. To address this issue, we propose a novel adversarial attack technique known as Adversarial Restoration (AdvRestore), which enhances both visual quality and transferability of adversarial face examples by leveraging a face restoration prior. In our approach, we initially train a Restoration Latent Diffusion Model (RLDM) designed for face restoration. Subsequently, we employ the inference process of RLDM to generate adversarial face examples. The adversarial perturbations are applied to the intermediate features of RLDM. Additionally, by treating RLDM face restoration as a sibling task, the transferability of the generated adversarial face examples is further improved. Our experimental results validate the effectiveness of the proposed attack method.



## **24. Attacking logo-based phishing website detectors with adversarial perturbations**

cs.CR

To appear in ESORICS 2023

**SubmitDate**: 2023-09-13    [abs](http://arxiv.org/abs/2308.09392v2) [paper-pdf](http://arxiv.org/pdf/2308.09392v2)

**Authors**: Jehyun Lee, Zhe Xin, Melanie Ng Pei See, Kanav Sabharwal, Giovanni Apruzzese, Dinil Mon Divakaran

**Abstract**: Recent times have witnessed the rise of anti-phishing schemes powered by deep learning (DL). In particular, logo-based phishing detectors rely on DL models from Computer Vision to identify logos of well-known brands on webpages, to detect malicious webpages that imitate a given brand. For instance, Siamese networks have demonstrated notable performance for these tasks, enabling the corresponding anti-phishing solutions to detect even "zero-day" phishing webpages. In this work, we take the next step of studying the robustness of logo-based phishing detectors against adversarial ML attacks. We propose a novel attack exploiting generative adversarial perturbations to craft "adversarial logos" that evade phishing detectors. We evaluate our attacks through: (i) experiments on datasets containing real logos, to evaluate the robustness of state-of-the-art phishing detectors; and (ii) user studies to gauge whether our adversarial logos can deceive human eyes. The results show that our proposed attack is capable of crafting perturbed logos subtle enough to evade various DL models-achieving an evasion rate of up to 95%. Moreover, users are not able to spot significant differences between generated adversarial logos and original ones.



## **25. Adversaries with Limited Information in the Friedkin--Johnsen Model**

cs.SI

KDD'23

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2306.10313v2) [paper-pdf](http://arxiv.org/pdf/2306.10313v2)

**Authors**: Sijing Tu, Stefan Neumann, Aristides Gionis

**Abstract**: In recent years, online social networks have been the target of adversaries who seek to introduce discord into societies, to undermine democracies and to destabilize communities. Often the goal is not to favor a certain side of a conflict but to increase disagreement and polarization. To get a mathematical understanding of such attacks, researchers use opinion-formation models from sociology, such as the Friedkin--Johnsen model, and formally study how much discord the adversary can produce when altering the opinions for only a small set of users. In this line of work, it is commonly assumed that the adversary has full knowledge about the network topology and the opinions of all users. However, the latter assumption is often unrealistic in practice, where user opinions are not available or simply difficult to estimate accurately.   To address this concern, we raise the following question: Can an attacker sow discord in a social network, even when only the network topology is known? We answer this question affirmatively. We present approximation algorithms for detecting a small set of users who are highly influential for the disagreement and polarization in the network. We show that when the adversary radicalizes these users and if the initial disagreement/polarization in the network is not very high, then our method gives a constant-factor approximation on the setting when the user opinions are known. To find the set of influential users, we provide a novel approximation algorithm for a variant of MaxCut in graphs with positive and negative edge weights. We experimentally evaluate our methods, which have access only to the network topology, and we find that they have similar performance as methods that have access to the network topology and all user opinions. We further present an NP-hardness proof, which was an open question by Chen and Racz [IEEE Trans. Netw. Sci. Eng., 2021].



## **26. Using Reed-Muller Codes for Classification with Rejection and Recovery**

cs.LG

38 pages, 7 figures

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2309.06359v1) [paper-pdf](http://arxiv.org/pdf/2309.06359v1)

**Authors**: Daniel Fentham, David Parker, Mark Ryan

**Abstract**: When deploying classifiers in the real world, users expect them to respond to inputs appropriately. However, traditional classifiers are not equipped to handle inputs which lie far from the distribution they were trained on. Malicious actors can exploit this defect by making adversarial perturbations designed to cause the classifier to give an incorrect output. Classification-with-rejection methods attempt to solve this problem by allowing networks to refuse to classify an input in which they have low confidence. This works well for strongly adversarial examples, but also leads to the rejection of weakly perturbed images, which intuitively could be correctly classified. To address these issues, we propose Reed-Muller Aggregation Networks (RMAggNet), a classifier inspired by Reed-Muller error-correction codes which can correct and reject inputs. This paper shows that RMAggNet can minimise incorrectness while maintaining good correctness over multiple adversarial attacks at different perturbation budgets by leveraging the ability to correct errors in the classification process. This provides an alternative classification-with-rejection method which can reduce the amount of additional processing in situations where a small number of incorrect classifications are permissible.



## **27. Inaudible Adversarial Perturbation: Manipulating the Recognition of User Speech in Real Time**

cs.CR

Accepted by NDSS Symposium 2024. Please cite this paper as "Xinfeng  Li, Chen Yan, Xuancun Lu, Zihan Zeng, Xiaoyu Ji, Wenyuan Xu. Inaudible  Adversarial Perturbation: Manipulating the Recognition of User Speech in Real  Time. In Network and Distributed System Security (NDSS) Symposium 2024."

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2308.01040v3) [paper-pdf](http://arxiv.org/pdf/2308.01040v3)

**Authors**: Xinfeng Li, Chen Yan, Xuancun Lu, Zihan Zeng, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Automatic speech recognition (ASR) systems have been shown to be vulnerable to adversarial examples (AEs). Recent success all assumes that users will not notice or disrupt the attack process despite the existence of music/noise-like sounds and spontaneous responses from voice assistants. Nonetheless, in practical user-present scenarios, user awareness may nullify existing attack attempts that launch unexpected sounds or ASR usage. In this paper, we seek to bridge the gap in existing research and extend the attack to user-present scenarios. We propose VRIFLE, an inaudible adversarial perturbation (IAP) attack via ultrasound delivery that can manipulate ASRs as a user speaks. The inherent differences between audible sounds and ultrasounds make IAP delivery face unprecedented challenges such as distortion, noise, and instability. In this regard, we design a novel ultrasonic transformation model to enhance the crafted perturbation to be physically effective and even survive long-distance delivery. We further enable VRIFLE's robustness by adopting a series of augmentation on user and real-world variations during the generation process. In this way, VRIFLE features an effective real-time manipulation of the ASR output from different distances and under any speech of users, with an alter-and-mute strategy that suppresses the impact of user disruption. Our extensive experiments in both digital and physical worlds verify VRIFLE's effectiveness under various configurations, robustness against six kinds of defenses, and universality in a targeted manner. We also show that VRIFLE can be delivered with a portable attack device and even everyday-life loudspeakers.



## **28. Adversarial Attacks Assessment of Salient Object Detection via Symbolic Learning**

cs.CV

14 pages, 8 figures, 6 tables, IEEE Transactions on Emerging Topics  in Computing, Accepted for publication

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2309.05900v1) [paper-pdf](http://arxiv.org/pdf/2309.05900v1)

**Authors**: Gustavo Olague, Roberto Pineda, Gerardo Ibarra-Vazquez, Matthieu Olague, Axel Martinez, Sambit Bakshi, Jonathan Vargas, Isnardo Reducindo

**Abstract**: Machine learning is at the center of mainstream technology and outperforms classical approaches to handcrafted feature design. Aside from its learning process for artificial feature extraction, it has an end-to-end paradigm from input to output, reaching outstandingly accurate results. However, security concerns about its robustness to malicious and imperceptible perturbations have drawn attention since its prediction can be changed entirely. Salient object detection is a research area where deep convolutional neural networks have proven effective but whose trustworthiness represents a significant issue requiring analysis and solutions to hackers' attacks. Brain programming is a kind of symbolic learning in the vein of good old-fashioned artificial intelligence. This work provides evidence that symbolic learning robustness is crucial in designing reliable visual attention systems since it can withstand even the most intense perturbations. We test this evolutionary computation methodology against several adversarial attacks and noise perturbations using standard databases and a real-world problem of a shorebird called the Snowy Plover portraying a visual attention task. We compare our methodology with five different deep learning approaches, proving that they do not match the symbolic paradigm regarding robustness. All neural networks suffer significant performance losses, while brain programming stands its ground and remains unaffected. Also, by studying the Snowy Plover, we remark on the importance of security in surveillance activities regarding wildlife protection and conservation.



## **29. Generalized Attacks on Face Verification Systems**

cs.CR

**SubmitDate**: 2023-09-12    [abs](http://arxiv.org/abs/2309.05879v1) [paper-pdf](http://arxiv.org/pdf/2309.05879v1)

**Authors**: Ehsan Nazari, Paula Branco, Guy-Vincent Jourdan

**Abstract**: Face verification (FV) using deep neural network models has made tremendous progress in recent years, surpassing human accuracy and seeing deployment in various applications such as border control and smartphone unlocking. However, FV systems are vulnerable to Adversarial Attacks, which manipulate input images to deceive these systems in ways usually unnoticeable to humans. This paper provides an in-depth study of attacks on FV systems. We introduce the DodgePersonation Attack that formulates the creation of face images that impersonate a set of given identities while avoiding being identified as any of the identities in a separate, disjoint set. A taxonomy is proposed to provide a unified view of different types of Adversarial Attacks against FV systems, including Dodging Attacks, Impersonation Attacks, and Master Face Attacks. Finally, we propose the ''One Face to Rule Them All'' Attack which implements the DodgePersonation Attack with state-of-the-art performance on a well-known scenario (Master Face Attack) and which can also be used for the new scenarios introduced in this paper. While the state-of-the-art Master Face Attack can produce a set of 9 images to cover 43.82% of the identities in their test database, with 9 images our attack can cover 57.27% to 58.5% of these identifies while giving the attacker the choice of the identity to use to create the impersonation. Moreover, the 9 generated attack images appear identical to a casual observer.



## **30. Robust Feature-Level Adversaries are Interpretability Tools**

cs.LG

NeurIPS 2022, code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2110.03605v7) [paper-pdf](http://arxiv.org/pdf/2110.03605v7)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstract**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore perceptible, interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying representations in models. Second, we show that these adversaries are uniquely versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results suggest that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations. Code is available at https://github.com/thestephencasper/feature_level_adv



## **31. Efficient Defense Against Model Stealing Attacks on Convolutional Neural Networks**

cs.LG

Accepted for publication at 2023 International Conference on Machine  Learning and Applications (ICMLA). Proceedings of ICMLA, Florida, USA  \c{opyright}2023 IEEE

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2309.01838v2) [paper-pdf](http://arxiv.org/pdf/2309.01838v2)

**Authors**: Kacem Khaled, Mouna Dhaouadi, Felipe Gohring de Magalhães, Gabriela Nicolescu

**Abstract**: Model stealing attacks have become a serious concern for deep learning models, where an attacker can steal a trained model by querying its black-box API. This can lead to intellectual property theft and other security and privacy risks. The current state-of-the-art defenses against model stealing attacks suggest adding perturbations to the prediction probabilities. However, they suffer from heavy computations and make impracticable assumptions about the adversary. They often require the training of auxiliary models. This can be time-consuming and resource-intensive which hinders the deployment of these defenses in real-world applications. In this paper, we propose a simple yet effective and efficient defense alternative. We introduce a heuristic approach to perturb the output probabilities. The proposed defense can be easily integrated into models without additional training. We show that our defense is effective in defending against three state-of-the-art stealing attacks. We evaluate our approach on large and quantized (i.e., compressed) Convolutional Neural Networks (CNNs) trained on several vision datasets. Our technique outperforms the state-of-the-art defenses with a $\times37$ faster inference latency without requiring any additional model and with a low impact on the model's performance. We validate that our defense is also effective for quantized CNNs targeting edge devices.



## **32. Byzantine Multiple Access Channels -- Part I: Reliable Communication**

cs.IT

This supercedes Part I of arxiv:1904.11925

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2211.12769v3) [paper-pdf](http://arxiv.org/pdf/2211.12769v3)

**Authors**: Neha Sangwan, Mayank Bakshi, Bikash Kumar Dey, Vinod M. Prabhakaran

**Abstract**: We study communication over a Multiple Access Channel (MAC) where users can possibly be adversarial. The receiver is unaware of the identity of the adversarial users (if any). When all users are non-adversarial, we want their messages to be decoded reliably. When a user behaves adversarially, we require that the honest users' messages be decoded reliably. An adversarial user can mount an attack by sending any input into the channel rather than following the protocol. It turns out that the $2$-user MAC capacity region follows from the point-to-point Arbitrarily Varying Channel (AVC) capacity. For the $3$-user MAC in which at most one user may be malicious, we characterize the capacity region for deterministic codes and randomized codes (where each user shares an independent random secret key with the receiver). These results are then generalized for the $k$-user MAC where the adversary may control all users in one out of a collection of given subsets.



## **33. Boosting Adversarial Transferability with Learnable Patch-wise Masks**

cs.CV

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2306.15931v2) [paper-pdf](http://arxiv.org/pdf/2306.15931v2)

**Authors**: Xingxing Wei, Shiji Zhao

**Abstract**: Adversarial examples have attracted widespread attention in security-critical applications because of their transferability across different models. Although many methods have been proposed to boost adversarial transferability, a gap still exists between capabilities and practical demand. In this paper, we argue that the model-specific discriminative regions are a key factor causing overfitting to the source model, and thus reducing the transferability to the target model. For that, a patch-wise mask is utilized to prune the model-specific regions when calculating adversarial perturbations. To accurately localize these regions, we present a learnable approach to automatically optimize the mask. Specifically, we simulate the target models in our framework, and adjust the patch-wise mask according to the feedback of the simulated models. To improve the efficiency, the differential evolutionary (DE) algorithm is utilized to search for patch-wise masks for a specific image. During iterative attacks, the learned masks are applied to the image to drop out the patches related to model-specific regions, thus making the gradients more generic and improving the adversarial transferability. The proposed approach is a preprocessing method and can be integrated with existing methods to further boost the transferability. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our method. We incorporate the proposed approach with existing methods to perform ensemble attacks and achieve an average success rate of 93.01% against seven advanced defense methods, which can effectively enhance the state-of-the-art transfer-based attack performance.



## **34. GIFD: A Generative Gradient Inversion Method with Feature Domain Optimization**

cs.CV

ICCV 2023

**SubmitDate**: 2023-09-11    [abs](http://arxiv.org/abs/2308.04699v2) [paper-pdf](http://arxiv.org/pdf/2308.04699v2)

**Authors**: Hao Fang, Bin Chen, Xuan Wang, Zhi Wang, Shu-Tao Xia

**Abstract**: Federated Learning (FL) has recently emerged as a promising distributed machine learning framework to preserve clients' privacy, by allowing multiple clients to upload the gradients calculated from their local data to a central server. Recent studies find that the exchanged gradients also take the risk of privacy leakage, e.g., an attacker can invert the shared gradients and recover sensitive data against an FL system by leveraging pre-trained generative adversarial networks (GAN) as prior knowledge. However, performing gradient inversion attacks in the latent space of the GAN model limits their expression ability and generalizability. To tackle these challenges, we propose \textbf{G}radient \textbf{I}nversion over \textbf{F}eature \textbf{D}omains (GIFD), which disassembles the GAN model and searches the feature domains of the intermediate layers. Instead of optimizing only over the initial latent code, we progressively change the optimized layer, from the initial latent space to intermediate layers closer to the output images. In addition, we design a regularizer to avoid unreal image generation by adding a small ${l_1}$ ball constraint to the searching range. We also extend GIFD to the out-of-distribution (OOD) setting, which weakens the assumption that the training sets of GANs and FL tasks obey the same data distribution. Extensive experiments demonstrate that our method can achieve pixel-level reconstruction and is superior to the existing methods. Notably, GIFD also shows great generalizability under different defense strategy settings and batch sizes.



## **35. Outlier Robust Adversarial Training**

cs.LG

Accepted by The 15th Asian Conference on Machine Learning (ACML 2023)

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2309.05145v1) [paper-pdf](http://arxiv.org/pdf/2309.05145v1)

**Authors**: Shu Hu, Zhenhuan Yang, Xin Wang, Yiming Ying, Siwei Lyu

**Abstract**: Supervised learning models are challenged by the intrinsic complexities of training data such as outliers and minority subpopulations and intentional attacks at inference time with adversarial samples. While traditional robust learning methods and the recent adversarial training approaches are designed to handle each of the two challenges, to date, no work has been done to develop models that are robust with regard to the low-quality training data and the potential adversarial attack at inference time simultaneously. It is for this reason that we introduce Outlier Robust Adversarial Training (ORAT) in this work. ORAT is based on a bi-level optimization formulation of adversarial training with a robust rank-based loss function. Theoretically, we show that the learning objective of ORAT satisfies the $\mathcal{H}$-consistency in binary classification, which establishes it as a proper surrogate to adversarial 0/1 loss. Furthermore, we analyze its generalization ability and provide uniform convergence rates in high probability. ORAT can be optimized with a simple algorithm. Experimental evaluations on three benchmark datasets demonstrate the effectiveness and robustness of ORAT in handling outliers and adversarial attacks. Our code is available at https://github.com/discovershu/ORAT.



## **36. DAD++: Improved Data-free Test Time Adversarial Defense**

cs.CV

IJCV Journal (Under Review)

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2309.05132v1) [paper-pdf](http://arxiv.org/pdf/2309.05132v1)

**Authors**: Gaurav Kumar Nayak, Inder Khatri, Shubham Randive, Ruchit Rawal, Anirban Chakraborty

**Abstract**: With the increasing deployment of deep neural networks in safety-critical applications such as self-driving cars, medical imaging, anomaly detection, etc., adversarial robustness has become a crucial concern in the reliability of these networks in real-world scenarios. A plethora of works based on adversarial training and regularization-based techniques have been proposed to make these deep networks robust against adversarial attacks. However, these methods require either retraining models or training them from scratch, making them infeasible to defend pre-trained models when access to training data is restricted. To address this problem, we propose a test time Data-free Adversarial Defense (DAD) containing detection and correction frameworks. Moreover, to further improve the efficacy of the correction framework in cases when the detector is under-confident, we propose a soft-detection scheme (dubbed as "DAD++"). We conduct a wide range of experiments and ablations on several datasets and network architectures to show the efficacy of our proposed approach. Furthermore, we demonstrate the applicability of our approach in imparting adversarial defense at test time under data-free (or data-efficient) applications/setups, such as Data-free Knowledge Distillation and Source-free Unsupervised Domain Adaptation, as well as Semi-supervised classification frameworks. We observe that in all the experiments and applications, our DAD++ gives an impressive performance against various adversarial attacks with a minimal drop in clean accuracy. The source code is available at: https://github.com/vcl-iisc/Improved-Data-free-Test-Time-Adversarial-Defense



## **37. Attacking c-MARL More Effectively: A Data Driven Approach**

cs.LG

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2202.03558v2) [paper-pdf](http://arxiv.org/pdf/2202.03558v2)

**Authors**: Nhan H. Pham, Lam M. Nguyen, Jie Chen, Hoang Thanh Lam, Subhro Das, Tsui-Wei Weng

**Abstract**: In recent years, a proliferation of methods were developed for cooperative multi-agent reinforcement learning (c-MARL). However, the robustness of c-MARL agents against adversarial attacks has been rarely explored. In this paper, we propose to evaluate the robustness of c-MARL agents via a model-based approach, named c-MBA. Our proposed formulation can craft much stronger adversarial state perturbations of c-MARL agents to lower total team rewards than existing model-free approaches. In addition, we propose the first victim-agent selection strategy and the first data-driven approach to define targeted failure states where each of them allows us to develop even stronger adversarial attack without the expert knowledge to the underlying environment. Our numerical experiments on two representative MARL benchmarks illustrate the advantage of our approach over other baselines: our model-based attack consistently outperforms other baselines in all tested environments.



## **38. Secure Set-Based State Estimation for Linear Systems under Adversarial Attacks on Sensors**

eess.SY

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2309.05075v1) [paper-pdf](http://arxiv.org/pdf/2309.05075v1)

**Authors**: Muhammad Umar B. Niazi, Michelle S. Chong, Amr Alanwar, Karl H. Johansson

**Abstract**: When a strategic adversary can attack multiple sensors of a system and freely choose a different set of sensors at different times, how can we ensure that the state estimate remains uncorrupted by the attacker? The existing literature addressing this problem mandates that the adversary can only corrupt less than half of the total number of sensors. This limitation is fundamental to all point-based secure state estimators because of their dependence on algorithms that rely on majority voting among sensors. However, in reality, an adversary with ample resources may not be limited to attacking less than half of the total number of sensors. This paper avoids the above-mentioned fundamental limitation by proposing a set-based approach that allows attacks on all but one sensor at any given time. We guarantee that the true state is always contained in the estimated set, which is represented by a collection of constrained zonotopes, provided that the system is bounded-input-bounded-state stable and redundantly observable via every combination of sensor subsets with size equal to the number of uncompromised sensors. Additionally, we show that the estimated set is secure and stable irrespective of the attack signals if the process and measurement noises are bounded. To detect the set of attacked sensors at each time, we propose a simple attack detection technique. However, we acknowledge that intelligently designed stealthy attacks may not be detected and, in the worst-case scenario, could even result in exponential growth in the algorithm's complexity. We alleviate this shortcoming by presenting a range of strategies that offer different levels of trade-offs between estimation performance and complexity.



## **39. Machine Translation Models Stand Strong in the Face of Adversarial Attacks**

cs.CL

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2309.06527v1) [paper-pdf](http://arxiv.org/pdf/2309.06527v1)

**Authors**: Pavel Burnyshev, Elizaveta Kostenok, Alexey Zaytsev

**Abstract**: Adversarial attacks expose vulnerabilities of deep learning models by introducing minor perturbations to the input, which lead to substantial alterations in the output. Our research focuses on the impact of such adversarial attacks on sequence-to-sequence (seq2seq) models, specifically machine translation models. We introduce algorithms that incorporate basic text perturbation heuristics and more advanced strategies, such as the gradient-based attack, which utilizes a differentiable approximation of the inherently non-differentiable translation metric. Through our investigation, we provide evidence that machine translation models display robustness displayed robustness against best performed known adversarial attacks, as the degree of perturbation in the output is directly proportional to the perturbation in the input. However, among underdogs, our attacks outperform alternatives, providing the best relative performance. Another strong candidate is an attack based on mixing of individual characters.



## **40. Mitigating Adversarial Attacks in Deepfake Detection: An Exploration of Perturbation and AI Techniques**

cs.LG

**SubmitDate**: 2023-09-10    [abs](http://arxiv.org/abs/2302.11704v2) [paper-pdf](http://arxiv.org/pdf/2302.11704v2)

**Authors**: Saminder Dhesi, Laura Fontes, Pedro Machado, Isibor Kennedy Ihianle, Farhad Fassihi Tash, David Ada Adama

**Abstract**: Deep learning constitutes a pivotal component within the realm of machine learning, offering remarkable capabilities in tasks ranging from image recognition to natural language processing. However, this very strength also renders deep learning models susceptible to adversarial examples, a phenomenon pervasive across a diverse array of applications. These adversarial examples are characterized by subtle perturbations artfully injected into clean images or videos, thereby causing deep learning algorithms to misclassify or produce erroneous outputs. This susceptibility extends beyond the confines of digital domains, as adversarial examples can also be strategically designed to target human cognition, leading to the creation of deceptive media, such as deepfakes. Deepfakes, in particular, have emerged as a potent tool to manipulate public opinion and tarnish the reputations of public figures, underscoring the urgent need to address the security and ethical implications associated with adversarial examples. This article delves into the multifaceted world of adversarial examples, elucidating the underlying principles behind their capacity to deceive deep learning algorithms. We explore the various manifestations of this phenomenon, from their insidious role in compromising model reliability to their impact in shaping the contemporary landscape of disinformation and misinformation. To illustrate progress in combating adversarial examples, we showcase the development of a tailored Convolutional Neural Network (CNN) designed explicitly to detect deepfakes, a pivotal step towards enhancing model robustness in the face of adversarial threats. Impressively, this custom CNN has achieved a precision rate of 76.2% on the DFDC dataset.



## **41. A Diamond Model Analysis on Twitter's Biggest Hack**

cs.CR

Discrepancies in the paper

**SubmitDate**: 2023-09-09    [abs](http://arxiv.org/abs/2306.15878v2) [paper-pdf](http://arxiv.org/pdf/2306.15878v2)

**Authors**: Chaitanya Rahalkar

**Abstract**: Cyberattacks have prominently increased over the past few years now, and have targeted actors from a wide variety of domains. Understanding the motivation, infrastructure, attack vectors, etc. behind such attacks is vital to proactively work against preventing such attacks in the future and also to analyze the economic and social impact of such attacks. In this paper, we leverage the diamond model to perform an intrusion analysis case study of the 2020 Twitter account hijacking Cyberattack. We follow this standardized incident response model to map the adversary, capability, infrastructure, and victim and perform a comprehensive analysis of the attack, and the impact posed by the attack from a Cybersecurity policy standpoint.



## **42. Good-looking but Lacking Faithfulness: Understanding Local Explanation Methods through Trend-based Testing**

cs.LG

**SubmitDate**: 2023-09-09    [abs](http://arxiv.org/abs/2309.05679v1) [paper-pdf](http://arxiv.org/pdf/2309.05679v1)

**Authors**: Jinwen He, Kai Chen, Guozhu Meng, Jiangshan Zhang, Congyi Li

**Abstract**: While enjoying the great achievements brought by deep learning (DL), people are also worried about the decision made by DL models, since the high degree of non-linearity of DL models makes the decision extremely difficult to understand. Consequently, attacks such as adversarial attacks are easy to carry out, but difficult to detect and explain, which has led to a boom in the research on local explanation methods for explaining model decisions. In this paper, we evaluate the faithfulness of explanation methods and find that traditional tests on faithfulness encounter the random dominance problem, \ie, the random selection performs the best, especially for complex data. To further solve this problem, we propose three trend-based faithfulness tests and empirically demonstrate that the new trend tests can better assess faithfulness than traditional tests on image, natural language and security tasks. We implement the assessment system and evaluate ten popular explanation methods. Benefiting from the trend tests, we successfully assess the explanation methods on complex data for the first time, bringing unprecedented discoveries and inspiring future research. Downstream tasks also greatly benefit from the tests. For example, model debugging equipped with faithful explanation methods performs much better for detecting and correcting accuracy and security problems.



## **43. Exploring Robust Features for Improving Adversarial Robustness**

cs.CV

12 pages, 8 figures

**SubmitDate**: 2023-09-09    [abs](http://arxiv.org/abs/2309.04650v1) [paper-pdf](http://arxiv.org/pdf/2309.04650v1)

**Authors**: Hong Wang, Yuefan Deng, Shinjae Yoo, Yuewei Lin

**Abstract**: While deep neural networks (DNNs) have revolutionized many fields, their fragility to carefully designed adversarial attacks impedes the usage of DNNs in safety-critical applications. In this paper, we strive to explore the robust features which are not affected by the adversarial perturbations, i.e., invariant to the clean image and its adversarial examples, to improve the model's adversarial robustness. Specifically, we propose a feature disentanglement model to segregate the robust features from non-robust features and domain specific features. The extensive experiments on four widely used datasets with different attacks demonstrate that robust features obtained from our model improve the model's adversarial robustness compared to the state-of-the-art approaches. Moreover, the trained domain discriminator is able to identify the domain specific features from the clean images and adversarial examples almost perfectly. This enables adversarial example detection without incurring additional computational costs. With that, we can also specify different classifiers for clean images and adversarial examples, thereby avoiding any drop in clean image accuracy.



## **44. Avoid Adversarial Adaption in Federated Learning by Multi-Metric Investigations**

cs.LG

25 pages, 14 figures, 23 tables, 11 equations

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2306.03600v2) [paper-pdf](http://arxiv.org/pdf/2306.03600v2)

**Authors**: Torsten Krauß, Alexandra Dmitrienko

**Abstract**: Federated Learning (FL) facilitates decentralized machine learning model training, preserving data privacy, lowering communication costs, and boosting model performance through diversified data sources. Yet, FL faces vulnerabilities such as poisoning attacks, undermining model integrity with both untargeted performance degradation and targeted backdoor attacks. Preventing backdoors proves especially challenging due to their stealthy nature.   Prominent mitigation techniques against poisoning attacks rely on monitoring certain metrics and filtering malicious model updates. While shown effective in evaluations, we argue that previous works didn't consider realistic real-world adversaries and data distributions. We define a new notion of strong adaptive adversaries, capable of adapting to multiple objectives simultaneously. Through extensive empirical tests, we show that existing defense methods can be easily circumvented in this adversary model. We also demonstrate, that existing defenses have limited effectiveness when no assumptions are made about underlying data distributions.   We introduce Metric-Cascades (MESAS), a novel defense method for more realistic scenarios and adversary models. MESAS employs multiple detection metrics simultaneously to identify poisoned model updates, creating a complex multi-objective optimization problem for adaptive attackers. In our extensive evaluation featuring nine backdoors and three datasets, MESAS consistently detects even strong adaptive attackers. Furthermore, MESAS outperforms existing defenses in distinguishing backdoors from data distribution-related distortions within and across clients. MESAS is the first defense robust against strong adaptive adversaries, effective in real-world data scenarios, with an average overhead of just 24.37 seconds.



## **45. Verifiable Learning for Robust Tree Ensembles**

cs.LG

19 pages, 5 figures; full version of the revised paper accepted at  ACM CCS 2023

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2305.03626v2) [paper-pdf](http://arxiv.org/pdf/2305.03626v2)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on public datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, at the cost of an acceptable loss of accuracy in the non-adversarial setting.



## **46. FIVA: Facial Image and Video Anonymization and Anonymization Defense**

cs.CV

Accepted to ICCVW 2023 - DFAD 2023

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2309.04228v1) [paper-pdf](http://arxiv.org/pdf/2309.04228v1)

**Authors**: Felix Rosberg, Eren Erdal Aksoy, Cristofer Englund, Fernando Alonso-Fernandez

**Abstract**: In this paper, we present a new approach for facial anonymization in images and videos, abbreviated as FIVA. Our proposed method is able to maintain the same face anonymization consistently over frames with our suggested identity-tracking and guarantees a strong difference from the original face. FIVA allows for 0 true positives for a false acceptance rate of 0.001. Our work considers the important security issue of reconstruction attacks and investigates adversarial noise, uniform noise, and parameter noise to disrupt reconstruction attacks. In this regard, we apply different defense and protection methods against these privacy threats to demonstrate the scalability of FIVA. On top of this, we also show that reconstruction attack models can be used for detection of deep fakes. Last but not least, we provide experimental results showing how FIVA can even enable face swapping, which is purely trained on a single target image.



## **47. Counterfactual Explanations via Locally-guided Sequential Algorithmic Recourse**

cs.LG

7 pages, 5 figures, 3 appendix pages

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2309.04211v1) [paper-pdf](http://arxiv.org/pdf/2309.04211v1)

**Authors**: Edward A. Small, Jeffrey N. Clark, Christopher J. McWilliams, Kacper Sokol, Jeffrey Chan, Flora D. Salim, Raul Santos-Rodriguez

**Abstract**: Counterfactuals operationalised through algorithmic recourse have become a powerful tool to make artificial intelligence systems explainable. Conceptually, given an individual classified as y -- the factual -- we seek actions such that their prediction becomes the desired class y' -- the counterfactual. This process offers algorithmic recourse that is (1) easy to customise and interpret, and (2) directly aligned with the goals of each individual. However, the properties of a "good" counterfactual are still largely debated; it remains an open challenge to effectively locate a counterfactual along with its corresponding recourse. Some strategies use gradient-driven methods, but these offer no guarantees on the feasibility of the recourse and are open to adversarial attacks on carefully created manifolds. This can lead to unfairness and lack of robustness. Other methods are data-driven, which mostly addresses the feasibility problem at the expense of privacy, security and secrecy as they require access to the entire training data set. Here, we introduce LocalFACE, a model-agnostic technique that composes feasible and actionable counterfactual explanations using locally-acquired information at each step of the algorithmic recourse. Our explainer preserves the privacy of users by only leveraging data that it specifically requires to construct actionable algorithmic recourse, and protects the model by offering transparency solely in the regions deemed necessary for the intervention.



## **48. Adversarial attacks on hybrid classical-quantum Deep Learning models for Histopathological Cancer Detection**

quant-ph

7 pages, 8 figures, 2 Tables

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2309.06377v1) [paper-pdf](http://arxiv.org/pdf/2309.06377v1)

**Authors**: Biswaraj Baral, Reek Majumdar, Bhavika Bhalgamiya, Taposh Dutta Roy

**Abstract**: We present an effective application of quantum machine learning in histopathological cancer detection. The study here emphasizes two primary applications of hybrid classical-quantum Deep Learning models. The first application is to build a classification model for histopathological cancer detection using the quantum transfer learning strategy. The second application is to test the performance of this model for various adversarial attacks. Rather than using a single transfer learning model, the hybrid classical-quantum models are tested using multiple transfer learning models, especially ResNet18, VGG-16, Inception-v3, and AlexNet as feature extractors and integrate it with several quantum circuit-based variational quantum circuits (VQC) with high expressibility. As a result, we provide a comparative analysis of classical models and hybrid classical-quantum transfer learning models for histopathological cancer detection under several adversarial attacks. We compared the performance accuracy of the classical model with the hybrid classical-quantum model using pennylane default quantum simulator. We also observed that for histopathological cancer detection under several adversarial attacks, Hybrid Classical-Quantum (HCQ) models provided better accuracy than classical image classification models.



## **49. Blades: A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning**

cs.CR

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2206.05359v2) [paper-pdf](http://arxiv.org/pdf/2206.05359v2)

**Authors**: Shenghui Li, Edith Ngai, Fanghua Ye, Li Ju, Tianru Zhang, Thiemo Voigt

**Abstract**: Federated learning (FL) facilitates distributed training across clients, safeguarding the privacy of their data. The inherent distributed structure of FL introduces vulnerabilities, especially from adversarial (Byzantine) clients aiming to skew local updates to their advantage. Despite the plethora of research focusing on Byzantine-resilient FL, the academic community has yet to establish a comprehensive benchmark suite, pivotal for impartial assessment and comparison of different techniques.   This paper investigates existing techniques in Byzantine-resilient FL and introduces an open-source benchmark suite for convenient and fair performance comparisons. Our investigation begins with a systematic study of Byzantine attack and defense strategies. Subsequently, we present \ours, a scalable, extensible, and easily configurable benchmark suite that supports researchers and developers in efficiently implementing and validating novel strategies against baseline algorithms in Byzantine-resilient FL. The design of \ours incorporates key characteristics derived from our systematic study, encompassing the attacker's capabilities and knowledge, defense strategy categories, and factors influencing robustness. Blades contains built-in implementations of representative attack and defense strategies and offers user-friendly interfaces for seamlessly integrating new ideas.



## **50. Node Injection for Class-specific Network Poisoning**

cs.LG

28 pages, 5 figures

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2301.12277v2) [paper-pdf](http://arxiv.org/pdf/2301.12277v2)

**Authors**: Ansh Kumar Sharma, Rahul Kukreja, Mayank Kharbanda, Tanmoy Chakraborty

**Abstract**: Graph Neural Networks (GNNs) are powerful in learning rich network representations that aid the performance of downstream tasks. However, recent studies showed that GNNs are vulnerable to adversarial attacks involving node injection and network perturbation. Among these, node injection attacks are more practical as they don't require manipulation in the existing network and can be performed more realistically. In this paper, we propose a novel problem statement - a class-specific poison attack on graphs in which the attacker aims to misclassify specific nodes in the target class into a different class using node injection. Additionally, nodes are injected in such a way that they camouflage as benign nodes. We propose NICKI, a novel attacking strategy that utilizes an optimization-based approach to sabotage the performance of GNN-based node classifiers. NICKI works in two phases - it first learns the node representation and then generates the features and edges of the injected nodes. Extensive experiments and ablation studies on four benchmark networks show that NICKI is consistently better than four baseline attacking strategies for misclassifying nodes in the target class. We also show that the injected nodes are properly camouflaged as benign, thus making the poisoned graph indistinguishable from its clean version w.r.t various topological properties.



