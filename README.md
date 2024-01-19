# Latest Adversarial Attack Papers
**update at 2024-01-19 11:40:53**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Marrying Adapters and Mixup to Efficiently Enhance the Adversarial Robustness of Pre-Trained Language Models for Text Classification**

cs.CL

10 pages and 2 figures

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10111v1) [paper-pdf](http://arxiv.org/pdf/2401.10111v1)

**Authors**: Tuc Nguyen, Thai Le

**Abstract**: Existing works show that augmenting training data of neural networks using both clean and adversarial examples can enhance their generalizability under adversarial attacks. However, this training approach often leads to performance degradation on clean inputs. Additionally, it requires frequent re-training of the entire model to account for new attack types, resulting in significant and costly computations. Such limitations make adversarial training mechanisms less practical, particularly for complex Pre-trained Language Models (PLMs) with millions or even billions of parameters. To overcome these challenges while still harnessing the theoretical benefits of adversarial training, this study combines two concepts: (1) adapters, which enable parameter-efficient fine-tuning, and (2) Mixup, which train NNs via convex combinations of pairs data pairs. Intuitively, we propose to fine-tune PLMs through convex combinations of non-data pairs of fine-tuned adapters, one trained with clean and another trained with adversarial examples. Our experiments show that the proposed method achieves the best trade-off between training efficiency and predictive performance, both with and without attacks compared to other baselines on a variety of downstream tasks.



## **2. Power in Numbers: Robust reading comprehension by finetuning with four adversarial sentences per example**

cs.CL

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10091v1) [paper-pdf](http://arxiv.org/pdf/2401.10091v1)

**Authors**: Ariel Marcus

**Abstract**: Recent models have achieved human level performance on the Stanford Question Answering Dataset when using F1 scores to evaluate the reading comprehension task. Yet, teaching machines to comprehend text has not been solved in the general case. By appending one adversarial sentence to the context paragraph, past research has shown that the F1 scores from reading comprehension models drop almost in half. In this paper, I replicate past adversarial research with a new model, ELECTRA-Small, and demonstrate that the new model's F1 score drops from 83.9% to 29.2%. To improve ELECTRA-Small's resistance to this attack, I finetune the model on SQuAD v1.1 training examples with one to five adversarial sentences appended to the context paragraph. Like past research, I find that the finetuned model on one adversarial sentence does not generalize well across evaluation datasets. However, when finetuned on four or five adversarial sentences the model attains an F1 score of more than 70% on most evaluation datasets with multiple appended and prepended adversarial sentences. The results suggest that with enough examples we can make models robust to adversarial attacks.



## **3. HGAttack: Transferable Heterogeneous Graph Adversarial Attack**

cs.LG

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09945v1) [paper-pdf](http://arxiv.org/pdf/2401.09945v1)

**Authors**: He Zhao, Zhiwei Zeng, Yongwei Wang, Deheng Ye, Chunyan Miao

**Abstract**: Heterogeneous Graph Neural Networks (HGNNs) are increasingly recognized for their performance in areas like the web and e-commerce, where resilience against adversarial attacks is crucial. However, existing adversarial attack methods, which are primarily designed for homogeneous graphs, fall short when applied to HGNNs due to their limited ability to address the structural and semantic complexity of HGNNs. This paper introduces HGAttack, the first dedicated gray box evasion attack method for heterogeneous graphs. We design a novel surrogate model to closely resemble the behaviors of the target HGNN and utilize gradient-based methods for perturbation generation. Specifically, the proposed surrogate model effectively leverages heterogeneous information by extracting meta-path induced subgraphs and applying GNNs to learn node embeddings with distinct semantics from each subgraph. This approach improves the transferability of generated attacks on the target HGNN and significantly reduces memory costs. For perturbation generation, we introduce a semantics-aware mechanism that leverages subgraph gradient information to autonomously identify vulnerable edges across a wide range of relations within a constrained perturbation budget. We validate HGAttack's efficacy with comprehensive experiments on three datasets, providing empirical analyses of its generated perturbations. Outperforming baseline methods, HGAttack demonstrated significant efficacy in diminishing the performance of target HGNN models, affirming the effectiveness of our approach in evaluating the robustness of HGNNs against adversarial attacks.



## **4. Universally Robust Graph Neural Networks by Preserving Neighbor Similarity**

cs.LG

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09754v1) [paper-pdf](http://arxiv.org/pdf/2401.09754v1)

**Authors**: Yulin Zhu, Yuni Lai, Xing Ai, Kai Zhou

**Abstract**: Despite the tremendous success of graph neural networks in learning relational data, it has been widely investigated that graph neural networks are vulnerable to structural attacks on homophilic graphs. Motivated by this, a surge of robust models is crafted to enhance the adversarial robustness of graph neural networks on homophilic graphs. However, the vulnerability based on heterophilic graphs remains a mystery to us. To bridge this gap, in this paper, we start to explore the vulnerability of graph neural networks on heterophilic graphs and theoretically prove that the update of the negative classification loss is negatively correlated with the pairwise similarities based on the powered aggregated neighbor features. This theoretical proof explains the empirical observations that the graph attacker tends to connect dissimilar node pairs based on the similarities of neighbor features instead of ego features both on homophilic and heterophilic graphs. In this way, we novelly introduce a novel robust model termed NSPGNN which incorporates a dual-kNN graphs pipeline to supervise the neighbor similarity-guided propagation. This propagation utilizes the low-pass filter to smooth the features of node pairs along the positive kNN graphs and the high-pass filter to discriminate the features of node pairs along the negative kNN graphs. Extensive experiments on both homophilic and heterophilic graphs validate the universal robustness of NSPGNN compared to the state-of-the-art methods.



## **5. Hijacking Attacks against Neural Networks by Analyzing Training Data**

cs.CR

Accepted by the 33rd USENIX Security Symposium (USENIX Security  2024); Full Version

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09740v1) [paper-pdf](http://arxiv.org/pdf/2401.09740v1)

**Authors**: Yunjie Ge, Qian Wang, Huayang Huang, Qi Li, Cong Wang, Chao Shen, Lingchen Zhao, Peipei Jiang, Zheng Fang, Shenyi Zhang

**Abstract**: Backdoors and adversarial examples are the two primary threats currently faced by deep neural networks (DNNs). Both attacks attempt to hijack the model behaviors with unintended outputs by introducing (small) perturbations to the inputs. Backdoor attacks, despite the high success rates, often require a strong assumption, which is not always easy to achieve in reality. Adversarial example attacks, which put relatively weaker assumptions on attackers, often demand high computational resources, yet do not always yield satisfactory success rates when attacking mainstream black-box models in the real world. These limitations motivate the following research question: can model hijacking be achieved more simply, with a higher attack success rate and more reasonable assumptions? In this paper, we propose CleanSheet, a new model hijacking attack that obtains the high performance of backdoor attacks without requiring the adversary to tamper with the model training process. CleanSheet exploits vulnerabilities in DNNs stemming from the training data. Specifically, our key idea is to treat part of the clean training data of the target model as "poisoned data," and capture the characteristics of these data that are more sensitive to the model (typically called robust features) to construct "triggers." These triggers can be added to any input example to mislead the target model, similar to backdoor attacks. We validate the effectiveness of CleanSheet through extensive experiments on 5 datasets, 79 normally trained models, 68 pruned models, and 39 defensive models. Results show that CleanSheet exhibits performance comparable to state-of-the-art backdoor attacks, achieving an average attack success rate (ASR) of 97.5% on CIFAR-100 and 92.4% on GTSRB, respectively. Furthermore, CleanSheet consistently maintains a high ASR, when confronted with various mainstream backdoor defenses.



## **6. X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network**

cs.CR

This is the Accepted version of an article for publication in IEEE  TVT

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2303.12278v2) [paper-pdf](http://arxiv.org/pdf/2303.12278v2)

**Authors**: Seonghoon Jeong, Sangho Lee, Hwejae Lee, Huy Kang Kim

**Abstract**: Controller Area Network (CAN) is an essential networking protocol that connects multiple electronic control units (ECUs) in a vehicle. However, CAN-based in-vehicle networks (IVNs) face security risks owing to the CAN mechanisms. An adversary can sabotage a vehicle by leveraging the security risks if they can access the CAN bus. Thus, recent actions and cybersecurity regulations (e.g., UNR 155) require carmakers to implement intrusion detection systems (IDSs) in their vehicles. The IDS should detect cyberattacks and provide additional information to analyze conducted attacks. Although many IDSs have been proposed, considerations regarding their feasibility and explainability remain lacking. This study proposes X-CANIDS, which is a novel IDS for CAN-based IVNs. X-CANIDS dissects the payloads in CAN messages into human-understandable signals using a CAN database. The signals improve the intrusion detection performance compared with the use of bit representations of raw payloads. These signals also enable an understanding of which signal or ECU is under attack. X-CANIDS can detect zero-day attacks because it does not require any labeled dataset in the training phase. We confirmed the feasibility of the proposed method through a benchmark test on an automotive-grade embedded device with a GPU. The results of this work will be valuable to carmakers and researchers considering the installation of in-vehicle IDSs for their vehicles.



## **7. Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack**

cs.CV

9 pages, 5 figures

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09673v1) [paper-pdf](http://arxiv.org/pdf/2401.09673v1)

**Authors**: Zhongliang Guo, Kaixuan Wang, Weiye Li, Yifei Qian, Ognjen Arandjelović, Lei Fang

**Abstract**: Neural style transfer (NST) is widely adopted in computer vision to generate new images with arbitrary styles. This process leverages neural networks to merge aesthetic elements of a style image with the structural aspects of a content image into a harmoniously integrated visual result. However, unauthorized NST can exploit artwork. Such misuse raises socio-technical concerns regarding artists' rights and motivates the development of technical approaches for the proactive protection of original creations. Adversarial attack is a concept primarily explored in machine learning security. Our work introduces this technique to protect artists' intellectual property. In this paper Locally Adaptive Adversarial Color Attack (LAACA), a method for altering images in a manner imperceptible to the human eyes but disruptive to NST. Specifically, we design perturbations targeting image areas rich in high-frequency content, generated by disrupting intermediate features. Our experiments and user study confirm that by attacking NST using the proposed method results in visually worse neural style transfer, thus making it an effective solution for visual artwork protection.



## **8. MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks**

eess.IV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09624v1) [paper-pdf](http://arxiv.org/pdf/2401.09624v1)

**Authors**: Giovanni Pasqualino, Luca Guarnera, Alessandro Ortis, Sebastiano Battiato

**Abstract**: The progress in generative models, particularly Generative Adversarial Networks (GANs), opened new possibilities for image generation but raised concerns about potential malicious uses, especially in sensitive areas like medical imaging. This study introduces MITS-GAN, a novel approach to prevent tampering in medical images, with a specific focus on CT scans. The approach disrupts the output of the attacker's CT-GAN architecture by introducing imperceptible but yet precise perturbations. Specifically, the proposed approach involves the introduction of appropriate Gaussian noise to the input as a protective measure against various attacks. Our method aims to enhance tamper resistance, comparing favorably to existing techniques. Experimental results on a CT scan dataset demonstrate MITS-GAN's superior performance, emphasizing its ability to generate tamper-resistant images with negligible artifacts. As image tampering in medical domains poses life-threatening risks, our proactive approach contributes to the responsible and ethical use of generative models. This work provides a foundation for future research in countering cyber threats in medical imaging. Models and codes are publicly available at the following link \url{https://iplab.dmi.unict.it/MITS-GAN-2024/}.



## **9. Towards Scalable and Robust Model Versioning**

cs.LG

Accepted in IEEE SaTML 2024

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09574v1) [paper-pdf](http://arxiv.org/pdf/2401.09574v1)

**Authors**: Wenxin Ding, Arjun Nitin Bhagoji, Ben Y. Zhao, Haitao Zheng

**Abstract**: As the deployment of deep learning models continues to expand across industries, the threat of malicious incursions aimed at gaining access to these deployed models is on the rise. Should an attacker gain access to a deployed model, whether through server breaches, insider attacks, or model inversion techniques, they can then construct white-box adversarial attacks to manipulate the model's classification outcomes, thereby posing significant risks to organizations that rely on these models for critical tasks. Model owners need mechanisms to protect themselves against such losses without the necessity of acquiring fresh training data - a process that typically demands substantial investments in time and capital.   In this paper, we explore the feasibility of generating multiple versions of a model that possess different attack properties, without acquiring new training data or changing model architecture. The model owner can deploy one version at a time and replace a leaked version immediately with a new version. The newly deployed model version can resist adversarial attacks generated leveraging white-box access to one or all previously leaked versions. We show theoretically that this can be accomplished by incorporating parameterized hidden distributions into the model training data, forcing the model to learn task-irrelevant features uniquely defined by the chosen data. Additionally, optimal choices of hidden distributions can produce a sequence of model versions capable of resisting compound transferability attacks over time. Leveraging our analytical insights, we design and implement a practical model versioning method for DNN classifiers, which leads to significant robustness improvements over existing methods. We believe our work presents a promising direction for safeguarding DNN services beyond their initial deployment.



## **10. Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability**

cs.CV

Accepted as a conference paper in NeurIPS'2023. Code repo:  https://github.com/xavihart/Diff-PGD

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2305.16494v3) [paper-pdf](http://arxiv.org/pdf/2305.16494v3)

**Authors**: Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen

**Abstract**: Neural networks are known to be susceptible to adversarial samples: small variations of natural examples crafted to deliberately mislead the models. While they can be easily generated using gradient-based techniques in digital and physical scenarios, they often differ greatly from the actual data distribution of natural images, resulting in a trade-off between strength and stealthiness. In this paper, we propose a novel framework dubbed Diffusion-Based Projected Gradient Descent (Diff-PGD) for generating realistic adversarial samples. By exploiting a gradient guided by a diffusion model, Diff-PGD ensures that adversarial samples remain close to the original data distribution while maintaining their effectiveness. Moreover, our framework can be easily customized for specific tasks such as digital attacks, physical-world attacks, and style-based attacks. Compared with existing methods for generating natural-style adversarial samples, our framework enables the separation of optimizing adversarial loss from other surrogate losses (e.g., content/smoothness/style loss), making it more stable and controllable. Finally, we demonstrate that the samples generated using Diff-PGD have better transferability and anti-purification power than traditional gradient-based methods. Code will be released in https://github.com/xavihart/Diff-PGD



## **11. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

cs.CV

under review

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.06637v3) [paper-pdf](http://arxiv.org/pdf/2401.06637v3)

**Authors**: Peter Lorenz, Ricard Durall, Janis Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.



## **12. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

cs.CV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2312.04960v2) [paper-pdf](http://arxiv.org/pdf/2312.04960v2)

**Authors**: Xiaoyun Xu, Shujian Yu, Jingzheng Wu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) achieve superior performance on various tasks compared to convolutional neural networks (CNNs), but ViTs are also vulnerable to adversarial attacks. Adversarial training is one of the most successful methods to build robust CNN models. Thus, recent works explored new methodologies for adversarial training of ViTs based on the differences between ViTs and CNNs, such as better training strategies, preventing attention from focusing on a single block, or discarding low-attention embeddings. However, these methods still follow the design of traditional supervised adversarial training, limiting the potential of adversarial training on ViTs. This paper proposes a novel defense method, MIMIR, which aims to build a different adversarial training methodology by utilizing Masked Image Modeling at pre-training. We create an autoencoder that accepts adversarial examples as input but takes the clean examples as the modeling target. Then, we create a mutual information (MI) penalty following the idea of the Information Bottleneck. Among the two information source inputs and corresponding adversarial perturbation, the perturbation information is eliminated due to the constraint of the modeling target. Next, we provide a theoretical analysis of MIMIR using the bounds of the MI penalty. We also design two adaptive attacks when the adversary is aware of the MIMIR defense and show that MIMIR still performs well. The experimental results show that MIMIR improves (natural and adversarial) accuracy on average by 4.19% on CIFAR-10 and 5.52% on ImageNet-1K, compared to baselines. On Tiny-ImageNet, we obtained improved natural accuracy of 2.99\% on average and comparable adversarial accuracy. Our code and trained models are publicly available https://github.com/xiaoyunxxy/MIMIR.



## **13. Username Squatting on Online Social Networks: A Study on X**

cs.CR

Accepted at ACM ASIA Conference on Computer and Communications  Security (AsiaCCS), 2024

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09209v1) [paper-pdf](http://arxiv.org/pdf/2401.09209v1)

**Authors**: Anastasios Lepipas, Anastasia Borovykh, Soteris Demetriou

**Abstract**: Adversaries have been targeting unique identifiers to launch typo-squatting, mobile app squatting and even voice squatting attacks. Anecdotal evidence suggest that online social networks (OSNs) are also plagued with accounts that use similar usernames. This can be confusing to users but can also be exploited by adversaries. However, to date no study characterizes this problem on OSNs. In this work, we define the username squatting problem and design the first multi-faceted measurement study to characterize it on X. We develop a username generation tool (UsernameCrazy) to help us analyze hundreds of thousands of username variants derived from celebrity accounts. Our study reveals that thousands of squatted usernames have been suspended by X, while tens of thousands that still exist on the network are likely bots. Out of these, a large number share similar profile pictures and profile names to the original account signalling impersonation attempts. We found that squatted accounts are being mentioned by mistake in tweets hundreds of thousands of times and are even being prioritized in searches by the network's search recommendation algorithm exacerbating the negative impact squatted accounts can have in OSNs. We use our insights and take the first step to address this issue by designing a framework (SQUAD) that combines UsernameCrazy with a new classifier to efficiently detect suspicious squatted accounts. Our evaluation of SQUAD's prototype implementation shows that it can achieve 94% F1-score when trained on a small dataset.



## **14. Attack and Reset for Unlearning: Exploiting Adversarial Noise toward Machine Unlearning through Parameter Re-initialization**

cs.LG

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08998v1) [paper-pdf](http://arxiv.org/pdf/2401.08998v1)

**Authors**: Yoonhwa Jung, Ikhyun Cho, Shun-Hsiang Hsu, Julia Hockenmaier

**Abstract**: With growing concerns surrounding privacy and regulatory compliance, the concept of machine unlearning has gained prominence, aiming to selectively forget or erase specific learned information from a trained model. In response to this critical need, we introduce a novel approach called Attack-and-Reset for Unlearning (ARU). This algorithm leverages meticulously crafted adversarial noise to generate a parameter mask, effectively resetting certain parameters and rendering them unlearnable. ARU outperforms current state-of-the-art results on two facial machine-unlearning benchmark datasets, MUFAC and MUCAC. In particular, we present the steps involved in attacking and masking that strategically filter and re-initialize network parameters biased towards the forget set. Our work represents a significant advancement in rendering data unexploitable to deep learning models through parameter re-initialization, achieved by harnessing adversarial noise to craft a mask.



## **15. A GAN-based data poisoning framework against anomaly detection in vertical federated learning**

cs.LG

6 pages, 7 figures. This work has been submitted to the IEEE for  possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08984v1) [paper-pdf](http://arxiv.org/pdf/2401.08984v1)

**Authors**: Xiaolin Chen, Daoguang Zan, Wei Li, Bei Guan, Yongji Wang

**Abstract**: In vertical federated learning (VFL), commercial entities collaboratively train a model while preserving data privacy. However, a malicious participant's poisoning attack may degrade the performance of this collaborative model. The main challenge in achieving the poisoning attack is the absence of access to the server-side top model, leaving the malicious participant without a clear target model. To address this challenge, we introduce an innovative end-to-end poisoning framework P-GAN. Specifically, the malicious participant initially employs semi-supervised learning to train a surrogate target model. Subsequently, this participant employs a GAN-based method to produce adversarial perturbations to degrade the surrogate target model's performance. Finally, the generator is obtained and tailored for VFL poisoning. Besides, we develop an anomaly detection algorithm based on a deep auto-encoder (DAE), offering a robust defense mechanism to VFL scenarios. Through extensive experiments, we evaluate the efficacy of P-GAN and DAE, and further analyze the factors that influence their performance.



## **16. RandOhm: Mitigating Impedance Side-channel Attacks using Randomized Circuit Configurations**

cs.CR

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08925v1) [paper-pdf](http://arxiv.org/pdf/2401.08925v1)

**Authors**: Saleh Khalaj Monfared, Domenic Forte, Shahin Tajik

**Abstract**: Physical side-channel attacks can compromise the security of integrated circuits. Most of the physical side-channel attacks (e.g., power or electromagnetic) exploit the dynamic behavior of a chip, typically manifesting as changes in current consumption or voltage fluctuations where algorithmic countermeasures, such as masking, can effectively mitigate the attacks. However, as demonstrated recently, these mitigation techniques are not entirely effective against backscattered side-channel attacks such as impedance analysis. In the case of an impedance attack, an adversary exploits the data-dependent impedance variations of chip power delivery network (PDN) to extract secret information. In this work, we introduce RandOhm, which exploits moving target defense (MTD) strategy based on partial reconfiguration of mainstream FPGAs, to defend against impedance side-channel attacks. We demonstrate that the information leakage through the PDN impedance could be reduced via run-time reconfiguration of the secret-sensitive parts of the circuitry. Hence, by constantly randomizing the placement and routing of the circuit, one can decorrelate the data-dependent computation from the impedance value. To validate our claims, we present a systematic approach equipped with two different partial reconfiguration strategies on implementations of the AES cipher realized on 28-nm FPGAs. We investigate the overhead of our mitigation in terms of delay and performance and provide security analysis by performing non-profiled and profiled impedance analysis attacks against these implementations to demonstrate the resiliency of our approach.



## **17. PPR: Enhancing Dodging Attacks while Maintaining Impersonation Attacks on Face Recognition Systems**

cs.CV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08903v1) [paper-pdf](http://arxiv.org/pdf/2401.08903v1)

**Authors**: Fengfan Zhou, Heifei Ling

**Abstract**: Adversarial Attacks on Face Recognition (FR) encompass two types: impersonation attacks and evasion attacks. We observe that achieving a successful impersonation attack on FR does not necessarily ensure a successful dodging attack on FR in the black-box setting. Introducing a novel attack method named Pre-training Pruning Restoration Attack (PPR), we aim to enhance the performance of dodging attacks whilst avoiding the degradation of impersonation attacks. Our method employs adversarial example pruning, enabling a portion of adversarial perturbations to be set to zero, while tending to maintain the attack performance. By utilizing adversarial example pruning, we can prune the pre-trained adversarial examples and selectively free up certain adversarial perturbations. Thereafter, we embed adversarial perturbations in the pruned area, which enhances the dodging performance of the adversarial face examples. The effectiveness of our proposed attack method is demonstrated through our experimental results, showcasing its superior performance.



## **18. Whispering Pixels: Exploiting Uninitialized Register Accesses in Modern GPUs**

cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08881v1) [paper-pdf](http://arxiv.org/pdf/2401.08881v1)

**Authors**: Frederik Dermot Pustelnik, Xhani Marvin Saß, Jean-Pierre Seifert

**Abstract**: Graphic Processing Units (GPUs) have transcended their traditional use-case of rendering graphics and nowadays also serve as a powerful platform for accelerating ubiquitous, non-graphical rendering tasks. One prominent task is inference of neural networks, which process vast amounts of personal data, such as audio, text or images. Thus, GPUs became integral components for handling vast amounts of potentially confidential data, which has awakened the interest of security researchers. This lead to the discovery of various vulnerabilities in GPUs in recent years. In this paper, we uncover yet another vulnerability class in GPUs: We found that some GPU implementations lack proper register initialization routines before shader execution, leading to unintended register content leakage of previously executed shader kernels. We showcase the existence of the aforementioned vulnerability on products of 3 major vendors - Apple, NVIDIA and Qualcomm. The vulnerability poses unique challenges to an adversary due to opaque scheduling and register remapping algorithms present in the GPU firmware, complicating the reconstruction of leaked data. In order to illustrate the real-world impact of this flaw, we showcase how these challenges can be solved for attacking various workloads on the GPU. First, we showcase how uninitialized registers leak arbitrary pixel data processed by fragment shaders. We further implement information leakage attacks on intermediate data of Convolutional Neural Networks (CNNs) and present the attack's capability to leak and reconstruct the output of Large Language Models (LLMs).



## **19. The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images**

cs.CV

ICLR 2024. Code:  https://github.com/mazurowski-lab/intrinsic-properties

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08865v1) [paper-pdf](http://arxiv.org/pdf/2401.08865v1)

**Authors**: Nicholas Konz, Maciej A. Mazurowski

**Abstract**: This paper investigates discrepancies in how neural networks learn from different imaging domains, which are commonly overlooked when adopting computer vision techniques from the domain of natural images to other specialized domains such as medical images. Recent works have found that the generalization error of a trained network typically increases with the intrinsic dimension ($d_{data}$) of its training set. Yet, the steepness of this relationship varies significantly between medical (radiological) and natural imaging domains, with no existing theoretical explanation. We address this gap in knowledge by establishing and empirically validating a generalization scaling law with respect to $d_{data}$, and propose that the substantial scaling discrepancy between the two considered domains may be at least partially attributed to the higher intrinsic "label sharpness" ($K_F$) of medical imaging datasets, a metric which we propose. Next, we demonstrate an additional benefit of measuring the label sharpness of a training set: it is negatively correlated with the trained model's adversarial robustness, which notably leads to models for medical images having a substantially higher vulnerability to adversarial attack. Finally, we extend our $d_{data}$ formalism to the related metric of learned representation intrinsic dimension ($d_{repr}$), derive a generalization scaling law with respect to $d_{repr}$, and show that $d_{data}$ serves as an upper bound for $d_{repr}$. Our theoretical results are supported by thorough experiments with six models and eleven natural and medical imaging datasets over a range of training set sizes. Our findings offer insights into the influence of intrinsic dataset properties on generalization, representation learning, and robustness in deep neural networks.



## **20. Game-Theoretic Neyman-Pearson Detection to Combat Strategic Evasion**

cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2206.05276v2) [paper-pdf](http://arxiv.org/pdf/2206.05276v2)

**Authors**: Yinan Hu, Quanyan Zhu

**Abstract**: The security in networked systems depends greatly on recognizing and identifying adversarial behaviors. Traditional detection methods focus on specific categories of attacks and have become inadequate for increasingly stealthy and deceptive attacks that are designed to bypass detection strategically. This work aims to develop a holistic theory to countermeasure such evasive attacks. We focus on extending a fundamental class of statistical-based detection methods based on Neyman-Pearson's (NP) hypothesis testing formulation. We propose game-theoretic frameworks to capture the conflicting relationship between a strategic evasive attacker and an evasion-aware NP detector. By analyzing both the equilibrium behaviors of the attacker and the NP detector, we characterize their performance using Equilibrium Receiver-Operational-Characteristic (EROC) curves. We show that the evasion-aware NP detectors outperform the passive ones in the way that the former can act strategically against the attacker's behavior and adaptively modify their decision rules based on the received messages. In addition, we extend our framework to a sequential setting where the user sends out identically distributed messages. We corroborate the analytical results with a case study of anomaly detection.



## **21. Benchmarking the Robustness of Image Watermarks**

cs.CV

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08573v1) [paper-pdf](http://arxiv.org/pdf/2401.08573v1)

**Authors**: Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, Furong Huang

**Abstract**: This paper investigates the weaknesses of image watermarking techniques. We present WAVES (Watermark Analysis Via Enhanced Stress-testing), a novel benchmark for assessing watermark robustness, overcoming the limitations of current evaluation methods.WAVES integrates detection and identification tasks, and establishes a standardized evaluation protocol comprised of a diverse range of stress tests. The attacks in WAVES range from traditional image distortions to advanced and novel variations of adversarial, diffusive, and embedding-based attacks. We introduce a normalized score of attack potency which incorporates several widely used image quality metrics and allows us to produce of an ordered ranking of attacks. Our comprehensive evaluation over reveals previously undetected vulnerabilities of several modern watermarking algorithms. WAVES is envisioned as a toolkit for the future development of robust watermarking systems.



## **22. Bag of Tricks to Boost Adversarial Transferability**

cs.CV

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08734v1) [paper-pdf](http://arxiv.org/pdf/2401.08734v1)

**Authors**: Zeliang Zhang, Rongyi Zhu, Wei Yao, Xiaosen Wang, Chenliang Xu

**Abstract**: Deep neural networks are widely known to be vulnerable to adversarial examples. However, vanilla adversarial examples generated under the white-box setting often exhibit low transferability across different models. Since adversarial transferability poses more severe threats to practical applications, various approaches have been proposed for better transferability, including gradient-based, input transformation-based, and model-related attacks, \etc. In this work, we find that several tiny changes in the existing adversarial attacks can significantly affect the attack performance, \eg, the number of iterations and step size. Based on careful studies of existing adversarial attacks, we propose a bag of tricks to enhance adversarial transferability, including momentum initialization, scheduled step size, dual example, spectral-based input transformation, and several ensemble strategies. Extensive experiments on the ImageNet dataset validate the high effectiveness of our proposed tricks and show that combining them can further boost adversarial transferability. Our work provides practical insights and techniques to enhance adversarial transferability, and offers guidance to improve the attack performance on the real-world application through simple adjustments.



## **23. SecPLF: Secure Protocols for Loanable Funds against Oracle Manipulation Attacks**

cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08520v1) [paper-pdf](http://arxiv.org/pdf/2401.08520v1)

**Authors**: Sanidhay Arora, Yingjiu Li, Yebo Feng, Jiahua Xu

**Abstract**: The evolving landscape of Decentralized Finance (DeFi) has raised critical security concerns, especially pertaining to Protocols for Loanable Funds (PLFs) and their dependency on price oracles, which are susceptible to manipulation. The emergence of flash loans has further amplified these risks, enabling increasingly complex oracle manipulation attacks that can lead to significant financial losses. Responding to this threat, we first dissect the attack mechanism by formalizing the standard operational and adversary models for PLFs. Based on our analysis, we propose SecPLF, a robust and practical solution designed to counteract oracle manipulation attacks efficiently. SecPLF operates by tracking a price state for each crypto-asset, including the recent price and the timestamp of its last update. By imposing price constraints on the price oracle usage, SecPLF ensures a PLF only engages a price oracle if the last recorded price falls within a defined threshold, thereby negating the profitability of potential attacks. Our evaluation based on historical market data confirms SecPLF's efficacy in providing high-confidence prevention against arbitrage attacks that arise due to minor price differences. SecPLF delivers proactive protection against oracle manipulation attacks, offering ease of implementation, oracle-agnostic property, and resource and cost efficiency.



## **24. Revealing Vulnerabilities in Stable Diffusion via Targeted Attacks**

cs.CV

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08725v1) [paper-pdf](http://arxiv.org/pdf/2401.08725v1)

**Authors**: Chenyu Zhang, Lanjun Wang, Anan Liu

**Abstract**: Recent developments in text-to-image models, particularly Stable Diffusion, have marked significant achievements in various applications. With these advancements, there are growing safety concerns about the vulnerability of the model that malicious entities exploit to generate targeted harmful images. However, the existing methods in the vulnerability of the model mainly evaluate the alignment between the prompt and generated images, but fall short in revealing the vulnerability associated with targeted image generation. In this study, we formulate the problem of targeted adversarial attack on Stable Diffusion and propose a framework to generate adversarial prompts. Specifically, we design a gradient-based embedding optimization method to craft reliable adversarial prompts that guide stable diffusion to generate specific images. Furthermore, after obtaining successful adversarial prompts, we reveal the mechanisms that cause the vulnerability of the model. Extensive experiments on two targeted attack tasks demonstrate the effectiveness of our method in targeted attacks. The code can be obtained in https://github.com/datar001/Revealing-Vulnerabilities-in-Stable-Diffusion-via-Targeted-Attacks.



## **25. A Generative Adversarial Attack for Multilingual Text Classifiers**

cs.CL

AAAI-24 Workshop on Artificial Intelligence for Cyber Security (AICS)

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08255v1) [paper-pdf](http://arxiv.org/pdf/2401.08255v1)

**Authors**: Tom Roth, Inigo Jauregi Unanue, Alsharif Abuadbba, Massimo Piccardi

**Abstract**: Current adversarial attack algorithms, where an adversary changes a text to fool a victim model, have been repeatedly shown to be effective against text classifiers. These attacks, however, generally assume that the victim model is monolingual and cannot be used to target multilingual victim models, a significant limitation given the increased use of these models. For this reason, in this work we propose an approach to fine-tune a multilingual paraphrase model with an adversarial objective so that it becomes able to generate effective adversarial examples against multilingual classifiers. The training objective incorporates a set of pre-trained models to ensure text quality and language consistency of the generated text. In addition, all the models are suitably connected to the generator by vocabulary-mapping matrices, allowing for full end-to-end differentiability of the overall training pipeline. The experimental validation over two multilingual datasets and five languages has shown the effectiveness of the proposed approach compared to existing baselines, particularly in terms of query efficiency. We also provide a detailed analysis of the generated attacks and discuss limitations and opportunities for future research.



## **26. FreqFed: A Frequency Analysis-Based Approach for Mitigating Poisoning Attacks in Federated Learning**

cs.CR

To appear in the Network and Distributed System Security (NDSS)  Symposium 2024. 16 pages, 8 figures, 12 tables, 1 algorithm, 3 equations

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2312.04432v2) [paper-pdf](http://arxiv.org/pdf/2312.04432v2)

**Authors**: Hossein Fereidooni, Alessandro Pegoraro, Phillip Rieger, Alexandra Dmitrienko, Ahmad-Reza Sadeghi

**Abstract**: Federated learning (FL) is a collaborative learning paradigm allowing multiple clients to jointly train a model without sharing their training data. However, FL is susceptible to poisoning attacks, in which the adversary injects manipulated model updates into the federated model aggregation process to corrupt or destroy predictions (untargeted poisoning) or implant hidden functionalities (targeted poisoning or backdoors). Existing defenses against poisoning attacks in FL have several limitations, such as relying on specific assumptions about attack types and strategies or data distributions or not sufficiently robust against advanced injection techniques and strategies and simultaneously maintaining the utility of the aggregated model. To address the deficiencies of existing defenses, we take a generic and completely different approach to detect poisoning (targeted and untargeted) attacks. We present FreqFed, a novel aggregation mechanism that transforms the model updates (i.e., weights) into the frequency domain, where we can identify the core frequency components that inherit sufficient information about weights. This allows us to effectively filter out malicious updates during local training on the clients, regardless of attack types, strategies, and clients' data distributions. We extensively evaluate the efficiency and effectiveness of FreqFed in different application domains, including image classification, word prediction, IoT intrusion detection, and speech recognition. We demonstrate that FreqFed can mitigate poisoning attacks effectively with a negligible impact on the utility of the aggregated model.



## **27. AdvSV: An Over-the-Air Adversarial Attack Dataset for Speaker Verification**

cs.SD

Accepted by ICASSP2024

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2310.05369v2) [paper-pdf](http://arxiv.org/pdf/2310.05369v2)

**Authors**: Li Wang, Jiaqi Li, Yuhao Luo, Jiahao Zheng, Lei Wang, Hao Li, Ke Xu, Chengfang Fang, Jie Shi, Zhizheng Wu

**Abstract**: It is known that deep neural networks are vulnerable to adversarial attacks. Although Automatic Speaker Verification (ASV) built on top of deep neural networks exhibits robust performance in controlled scenarios, many studies confirm that ASV is vulnerable to adversarial attacks. The lack of a standard dataset is a bottleneck for further research, especially reproducible research. In this study, we developed an open-source adversarial attack dataset for speaker verification research. As an initial step, we focused on the over-the-air attack. An over-the-air adversarial attack involves a perturbation generation algorithm, a loudspeaker, a microphone, and an acoustic environment. The variations in the recording configurations make it very challenging to reproduce previous research. The AdvSV dataset is constructed using the Voxceleb1 Verification test set as its foundation. This dataset employs representative ASV models subjected to adversarial attacks and records adversarial samples to simulate over-the-air attack settings. The scope of the dataset can be easily extended to include more types of adversarial attacks. The dataset will be released to the public under the CC BY-SA 4.0. In addition, we also provide a detection baseline for reproducible research.



## **28. IoTWarden: A Deep Reinforcement Learning Based Real-time Defense System to Mitigate Trigger-action IoT Attacks**

cs.CR

2024 IEEE Wireless Communications and Networking Conference (WCNC  2024)

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08141v1) [paper-pdf](http://arxiv.org/pdf/2401.08141v1)

**Authors**: Md Morshed Alam, Israt Jahan, Weichao Wang

**Abstract**: In trigger-action IoT platforms, IoT devices report event conditions to IoT hubs notifying their cyber states and let the hubs invoke actions in other IoT devices based on functional dependencies defined as rules in a rule engine. These functional dependencies create a chain of interactions that help automate network tasks. Adversaries exploit this chain to report fake event conditions to IoT hubs and perform remote injection attacks upon a smart environment to indirectly control targeted IoT devices. Existing defense efforts usually depend on static analysis over IoT apps to develop rule-based anomaly detection mechanisms. We also see ML-based defense mechanisms in the literature that harness physical event fingerprints to determine anomalies in an IoT network. However, these methods often demonstrate long response time and lack of adaptability when facing complicated attacks. In this paper, we propose to build a deep reinforcement learning based real-time defense system for injection attacks. We define the reward functions for defenders and implement a deep Q-network based approach to identify the optimal defense policy. Our experiments show that the proposed mechanism can effectively and accurately identify and defend against injection attacks with reasonable computation overhead.



## **29. Framework and Classification of Indicator of Compromise for physics-based attacks**

cs.CR

Pre-print is submitted to 2024 IEEE World Forum on Public Safety  Technology, and is under review

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08127v1) [paper-pdf](http://arxiv.org/pdf/2401.08127v1)

**Authors**: Vincent Tan

**Abstract**: Quantum communications are based on the law of physics for information security and the implications for this form of future information security enabled by quantum science has to be studied. Physics-based vulnerabilities may exist due to the inherent physics properties and behavior of quantum technologies such as Quantum Key Distribution (QKD), thus resulting in new threats that may emerge with attackers exploiting the physics-based vulnerabilities. There were many studies and experiments done to demonstrate the threat of physics-based attacks on quantum links. However, there is a lack of a framework that provides a common language to communicate about the threats and type of adversaries being dealt with for physics-based attacks. This paper is a review of physics-based attacks that were being investigated and attempt to initialize a framework based on the attack objectives and methodologies, referencing the concept from the well-established MITRE ATT&CK, therefore pioneering the classification of Indicator of Compromises (IoCs) for physics-based attacks. This paper will then pave the way for future work in the development of a forensic tool for the different classification of IoCs, with the methods of evidence collections and possible points of extractions for analysis being further investigated.



## **30. MGTBench: Benchmarking Machine-Generated Text Detection**

cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2303.14822v3) [paper-pdf](http://arxiv.org/pdf/2303.14822v3)

**Authors**: Xinlei He, Xinyue Shen, Zeyuan Chen, Michael Backes, Yang Zhang

**Abstract**: Nowadays, powerful large language models (LLMs) such as ChatGPT have demonstrated revolutionary power in a variety of tasks. Consequently, the detection of machine-generated texts (MGTs) is becoming increasingly crucial as LLMs become more advanced and prevalent. These models have the ability to generate human-like language, making it challenging to discern whether a text is authored by a human or a machine. This raises concerns regarding authenticity, accountability, and potential bias. However, existing methods for detecting MGTs are evaluated using different model architectures, datasets, and experimental settings, resulting in a lack of a comprehensive evaluation framework that encompasses various methodologies. Furthermore, it remains unclear how existing detection methods would perform against powerful LLMs. In this paper, we fill this gap by proposing the first benchmark framework for MGT detection against powerful LLMs, named MGTBench. Extensive evaluations on public datasets with curated texts generated by various powerful LLMs such as ChatGPT-turbo and Claude demonstrate the effectiveness of different detection methods. Our ablation study shows that a larger number of words in general leads to better performance and most detection methods can achieve similar performance with much fewer training samples. Moreover, we delve into a more challenging task: text attribution. Our findings indicate that the model-based detection methods still perform well in the text attribution task. To investigate the robustness of different detection methods, we consider three adversarial attacks, namely paraphrasing, random spacing, and adversarial perturbations. We discover that these attacks can significantly diminish detection effectiveness, underscoring the critical need for the development of more robust detection methods.



## **31. Towards Robust Neural Networks via Orthogonal Diversity**

cs.CV

accepted by Pattern Recognition

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2010.12190v5) [paper-pdf](http://arxiv.org/pdf/2010.12190v5)

**Authors**: Kun Fang, Qinghua Tao, Yingwen Wu, Tao Li, Jia Cai, Feipeng Cai, Xiaolin Huang, Jie Yang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to invisible perturbations on the images generated by adversarial attacks, which raises researches on the adversarial robustness of DNNs. A series of methods represented by the adversarial training and its variants have proven as one of the most effective techniques in enhancing the DNN robustness. Generally, adversarial training focuses on enriching the training data by involving perturbed data. Such data augmentation effect of the involved perturbed data in adversarial training does not contribute to the robustness of DNN itself and usually suffers from clean accuracy drop. Towards the robustness of DNN itself, we in this paper propose a novel defense that aims at augmenting the model in order to learn features that are adaptive to diverse inputs, including adversarial examples. More specifically, to augment the model, multiple paths are embedded into the network, and an orthogonality constraint is imposed on these paths to guarantee the diversity among them. A margin-maximization loss is then designed to further boost such DIversity via Orthogonality (DIO). In this way, the proposed DIO augments the model and enhances the robustness of DNN itself as the learned features can be corrected by these mutually-orthogonal paths. Extensive empirical results on various data sets, structures and attacks verify the stronger adversarial robustness of the proposed DIO utilizing model augmentation. Besides, DIO can also be flexibly combined with different data augmentation techniques (e.g., TRADES and DDPM), further promoting robustness gains.



## **32. Robustness Against Adversarial Attacks via Learning Confined Adversarial Polytopes**

cs.LG

The paper has been accepted in ICASSP 2024

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07991v1) [paper-pdf](http://arxiv.org/pdf/2401.07991v1)

**Authors**: Shayan Mohajer Hamidi, Linfeng Ye

**Abstract**: Deep neural networks (DNNs) could be deceived by generating human-imperceptible perturbations of clean samples. Therefore, enhancing the robustness of DNNs against adversarial attacks is a crucial task. In this paper, we aim to train robust DNNs by limiting the set of outputs reachable via a norm-bounded perturbation added to a clean sample. We refer to this set as adversarial polytope, and each clean sample has a respective adversarial polytope. Indeed, if the respective polytopes for all the samples are compact such that they do not intersect the decision boundaries of the DNN, then the DNN is robust against adversarial samples. Hence, the inner-working of our algorithm is based on learning \textbf{c}onfined \textbf{a}dversarial \textbf{p}olytopes (CAP). By conducting a thorough set of experiments, we demonstrate the effectiveness of CAP over existing adversarial robustness methods in improving the robustness of models against state-of-the-art attacks including AutoAttack.



## **33. Learning to Unlearn: Instance-wise Unlearning for Pre-trained Classifiers**

cs.LG

AAAI 2024 camera ready version

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2301.11578v3) [paper-pdf](http://arxiv.org/pdf/2301.11578v3)

**Authors**: Sungmin Cha, Sungjun Cho, Dasol Hwang, Honglak Lee, Taesup Moon, Moontae Lee

**Abstract**: Since the recent advent of regulations for data protection (e.g., the General Data Protection Regulation), there has been increasing demand in deleting information learned from sensitive data in pre-trained models without retraining from scratch. The inherent vulnerability of neural networks towards adversarial attacks and unfairness also calls for a robust method to remove or correct information in an instance-wise fashion, while retaining the predictive performance across remaining data. To this end, we consider instance-wise unlearning, of which the goal is to delete information on a set of instances from a pre-trained model, by either misclassifying each instance away from its original prediction or relabeling the instance to a different label. We also propose two methods that reduce forgetting on the remaining data: 1) utilizing adversarial examples to overcome forgetting at the representation-level and 2) leveraging weight importance metrics to pinpoint network parameters guilty of propagating unwanted information. Both methods only require the pre-trained model and data instances to forget, allowing painless application to real-life settings where the entire training set is unavailable. Through extensive experimentation on various image classification benchmarks, we show that our approach effectively preserves knowledge of remaining data while unlearning given instances in both single-task and continual unlearning scenarios.



## **34. ADMIn: Attacks on Dataset, Model and Input. A Threat Model for AI Based Software**

cs.CR

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07960v1) [paper-pdf](http://arxiv.org/pdf/2401.07960v1)

**Authors**: Vimal Kumar, Juliette Mayo, Khadija Bahiss

**Abstract**: Machine learning (ML) and artificial intelligence (AI) techniques have now become commonplace in software products and services. When threat modelling a system, it is therefore important that we consider threats unique to ML and AI techniques, in addition to threats to our software. In this paper, we present a threat model that can be used to systematically uncover threats to AI based software. The threat model consists of two main parts, a model of the software development process for AI based software and an attack taxonomy that has been developed using attacks found in adversarial AI research. We apply the threat model to two real life AI based software and discuss the process and the threats found.



## **35. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

cs.CL

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07867v1) [paper-pdf](http://arxiv.org/pdf/2401.07867v1)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of latest Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause detection evasion in all tested languages, where homoglyph attacks are especially successful.



## **36. Predominant Aspects on Security for Quantum Machine Learning: Literature Review**

quant-ph

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07774v1) [paper-pdf](http://arxiv.org/pdf/2401.07774v1)

**Authors**: Nicola Franco, Alona Sakhnenko, Leon Stolpmann, Daniel Thuerck, Fabian Petsch, Annika Rüll, Jeanette Miriam Lorenz

**Abstract**: Quantum Machine Learning (QML) has emerged as a promising intersection of quantum computing and classical machine learning, anticipated to drive breakthroughs in computational tasks. This paper discusses the question which security concerns and strengths are connected to QML by means of a systematic literature review. We categorize and review the security of QML models, their vulnerabilities inherent to quantum architectures, and the mitigation strategies proposed. The survey reveals that while QML possesses unique strengths, it also introduces novel attack vectors not seen in classical systems. Techniques like adversarial training, quantum noise exploitation, and quantum differential privacy have shown potential in enhancing QML robustness. Our review discuss the need for continued and rigorous research to ensure the secure deployment of QML in real-world applications. This work serves as a foundational reference for researchers and practitioners aiming to navigate the security aspects of QML.



## **37. Uncertainty-based Detection of Adversarial Attacks in Semantic Segmentation**

cs.CV

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2305.12825v2) [paper-pdf](http://arxiv.org/pdf/2305.12825v2)

**Authors**: Kira Maag, Asja Fischer

**Abstract**: State-of-the-art deep neural networks have proven to be highly powerful in a broad range of tasks, including semantic image segmentation. However, these networks are vulnerable against adversarial attacks, i.e., non-perceptible perturbations added to the input image causing incorrect predictions, which is hazardous in safety-critical applications like automated driving. Adversarial examples and defense strategies are well studied for the image classification task, while there has been limited research in the context of semantic segmentation. First works however show that the segmentation outcome can be severely distorted by adversarial attacks. In this work, we introduce an uncertainty-based approach for the detection of adversarial attacks in semantic segmentation. We observe that uncertainty as for example captured by the entropy of the output distribution behaves differently on clean and perturbed images and leverage this property to distinguish between the two cases. Our method works in a light-weight and post-processing manner, i.e., we do not modify the model or need knowledge of the process used for generating adversarial examples. In a thorough empirical analysis, we demonstrate the ability of our approach to detect perturbed images across multiple types of adversarial attacks.



## **38. Physics-constrained Attack against Convolution-based Human Motion Prediction**

cs.CV

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2306.11990v3) [paper-pdf](http://arxiv.org/pdf/2306.11990v3)

**Authors**: Chengxu Duan, Zhicheng Zhang, Xiaoli Liu, Yonghao Dang, Jianqin Yin

**Abstract**: Human motion prediction has achieved a brilliant performance with the help of convolution-based neural networks. However, currently, there is no work evaluating the potential risk in human motion prediction when facing adversarial attacks. The adversarial attack will encounter problems against human motion prediction in naturalness and data scale. To solve the problems above, we propose a new adversarial attack method that generates the worst-case perturbation by maximizing the human motion predictor's prediction error with physical constraints. Specifically, we introduce a novel adaptable scheme that facilitates the attack to suit the scale of the target pose and two physical constraints to enhance the naturalness of the adversarial example. The evaluating experiments on three datasets show that the prediction errors of all target models are enlarged significantly, which means current convolution-based human motion prediction models are vulnerable to the proposed attack. Based on the experimental results, we provide insights on how to enhance the adversarial robustness of the human motion predictor and how to improve the adversarial attack against human motion prediction.



## **39. Impartial Games: A Challenge for Reinforcement Learning**

cs.LG

**SubmitDate**: 2024-01-14    [abs](http://arxiv.org/abs/2205.12787v4) [paper-pdf](http://arxiv.org/pdf/2205.12787v4)

**Authors**: Bei Zhou, Søren Riis

**Abstract**: While AlphaZero-style reinforcement learning (RL) algorithms excel in various board games, in this paper we show that they face challenges on impartial games where players share pieces. We present a concrete example of a game - namely the children's game of Nim - and other impartial games that seem to be a stumbling block for AlphaZero-style and similar self-play reinforcement learning algorithms.   Our work is built on the challenges posed by the intricacies of data distribution on the ability of neural networks to learn parity functions, exacerbated by the noisy labels issue. Our findings are consistent with recent studies showing that AlphaZero-style algorithms are vulnerable to adversarial attacks and adversarial perturbations, showing the difficulty of learning to master the games in all legal states.   We show that Nim can be learned on small boards, but the learning progress of AlphaZero-style algorithms dramatically slows down when the board size increases. Intuitively, the difference between impartial games like Nim and partisan games like Chess and Go can be explained by the fact that if a small part of the board is covered for impartial games it is typically not possible to predict whether the position is won or lost as there is often zero correlation between the visible part of a partly blanked-out position and its correct evaluation. This situation starkly contrasts partisan games where a partly blanked-out board position typically provides abundant or at least non-trifle information about the value of the fully uncovered position.



## **40. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

cs.CR

11 pages, 5 figures

**SubmitDate**: 2024-01-14    [abs](http://arxiv.org/abs/2401.07261v1) [paper-pdf](http://arxiv.org/pdf/2401.07261v1)

**Authors**: Shoupeng Ren, Tianyu Tu, Jian Liu, Di Wu, Kui Ren

**Abstract**: DeFi incidents stemming from various smart contract vulnerabilities have culminated in financial damages exceeding 3 billion USD. The attacks causing such incidents commonly commence with the deployment of adversarial contracts, subsequently leveraging these contracts to execute adversarial transactions that exploit vulnerabilities in victim contracts. Existing defense mechanisms leverage heuristic or machine learning algorithms to detect adversarial transactions, but they face significant challenges in detecting private adversarial transactions. Namely, attackers can send adversarial transactions directly to miners, evading visibility within the blockchain network and effectively bypassing the detection. In this paper, we propose a new direction for detecting DeFi attacks, i.e., detecting adversarial contracts instead of adversarial transactions, allowing us to proactively identify potential attack intentions, even if they employ private adversarial transactions. Specifically, we observe that most adversarial contracts follow a similar pattern, e.g., anonymous fund source, closed-source, frequent token-related function calls. Based on this observation, we build a machine learning classifier that can effectively distinguish adversarial contracts from benign ones. We build a dataset consists of features extracted from 304 adversarial contracts and 13,000 benign contracts. Based on this dataset, we evaluate different classifiers, the results of which show that our method for identifying DeFi adversarial contracts performs exceptionally well. For example, the F1-Score for LightGBM-based classifier is 0.9434, with a remarkably low false positive rate of only 0.12%.



## **41. Crafter: Facial Feature Crafting against Inversion-based Identity Theft on Deep Models**

cs.CR

**SubmitDate**: 2024-01-14    [abs](http://arxiv.org/abs/2401.07205v1) [paper-pdf](http://arxiv.org/pdf/2401.07205v1)

**Authors**: Shiming Wang, Zhe Ji, Liyao Xiang, Hao Zhang, Xinbing Wang, Chenghu Zhou, Bo Li

**Abstract**: With the increased capabilities at the edge (e.g., mobile device) and more stringent privacy requirement, it becomes a recent trend for deep learning-enabled applications to pre-process sensitive raw data at the edge and transmit the features to the backend cloud for further processing. A typical application is to run machine learning (ML) services on facial images collected from different individuals. To prevent identity theft, conventional methods commonly rely on an adversarial game-based approach to shed the identity information from the feature. However, such methods can not defend against adaptive attacks, in which an attacker takes a countermove against a known defence strategy. We propose Crafter, a feature crafting mechanism deployed at the edge, to protect the identity information from adaptive model inversion attacks while ensuring the ML tasks are properly carried out in the cloud. The key defence strategy is to mislead the attacker to a non-private prior from which the attacker gains little about the private identity. In this case, the crafted features act like poison training samples for attackers with adaptive model updates. Experimental results indicate that Crafter successfully defends both basic and possible adaptive attacks, which can not be achieved by state-of-the-art adversarial game-based methods.



## **42. Experimental quantum e-commerce**

quant-ph

19 pages, 5 figures, 5 tables

**SubmitDate**: 2024-01-14    [abs](http://arxiv.org/abs/2308.08821v2) [paper-pdf](http://arxiv.org/pdf/2308.08821v2)

**Authors**: Xiao-Yu Cao, Bing-Hong Li, Yang Wang, Yao Fu, Hua-Lei Yin, Zeng-Bing Chen

**Abstract**: E-commerce, a type of trading that occurs at a high frequency on the Internet, requires guaranteeing the integrity, authentication and non-repudiation of messages through long distance. As current e-commerce schemes are vulnerable to computational attacks, quantum cryptography, ensuring information-theoretic security against adversary's repudiation and forgery, provides a solution to this problem. However, quantum solutions generally have much lower performance compared to classical ones. Besides, when considering imperfect devices, the performance of quantum schemes exhibits a significant decline. Here, for the first time, we demonstrate the whole e-commerce process of involving the signing of a contract and payment among three parties by proposing a quantum e-commerce scheme, which shows resistance of attacks from imperfect devices. Results show that with a maximum attenuation of 25 dB among participants, our scheme can achieve a signature rate of 0.82 times per second for an agreement size of approximately 0.428 megabit. This proposed scheme presents a promising solution for providing information-theoretic security for e-commerce.



## **43. Left-right Discrepancy for Adversarial Attack on Stereo Networks**

cs.CV

**SubmitDate**: 2024-01-14    [abs](http://arxiv.org/abs/2401.07188v1) [paper-pdf](http://arxiv.org/pdf/2401.07188v1)

**Authors**: Pengfei Wang, Xiaofei Hui, Beijia Lu, Nimrod Lilith, Jun Liu, Sameer Alam

**Abstract**: Stereo matching neural networks often involve a Siamese structure to extract intermediate features from left and right images. The similarity between these intermediate left-right features significantly impacts the accuracy of disparity estimation. In this paper, we introduce a novel adversarial attack approach that generates perturbation noise specifically designed to maximize the discrepancy between left and right image features. Extensive experiments demonstrate the superior capability of our method to induce larger prediction errors in stereo neural networks, e.g. outperforming existing state-of-the-art attack methods by 219% MAE on the KITTI dataset and 85% MAE on the Scene Flow dataset. Additionally, we extend our approach to include a proxy network black-box attack method, eliminating the need for access to stereo neural network. This method leverages an arbitrary network from a different vision task as a proxy to generate adversarial noise, effectively causing the stereo network to produce erroneous predictions. Our findings highlight a notable sensitivity of stereo networks to discrepancies in shallow layer features, offering valuable insights that could guide future research in enhancing the robustness of stereo vision systems.



## **44. Exploring Adversarial Attacks against Latent Diffusion Model from the Perspective of Adversarial Transferability**

cs.CV

24 pages, 13 figures

**SubmitDate**: 2024-01-13    [abs](http://arxiv.org/abs/2401.07087v1) [paper-pdf](http://arxiv.org/pdf/2401.07087v1)

**Authors**: Junxi Chen, Junhao Dong, Xiaohua Xie

**Abstract**: Recently, many studies utilized adversarial examples (AEs) to raise the cost of malicious image editing and copyright violation powered by latent diffusion models (LDMs). Despite their successes, a few have studied the surrogate model they used to generate AEs. In this paper, from the perspective of adversarial transferability, we investigate how the surrogate model's property influences the performance of AEs for LDMs. Specifically, we view the time-step sampling in the Monte-Carlo-based (MC-based) adversarial attack as selecting surrogate models. We find that the smoothness of surrogate models at different time steps differs, and we substantially improve the performance of the MC-based AEs by selecting smoother surrogate models. In the light of the theoretical framework on adversarial transferability in image classification, we also conduct a theoretical analysis to explain why smooth surrogate models can also boost AEs for LDMs.



## **45. Enhancing targeted transferability via feature space fine-tuning**

cs.CV

9 pages, 10 figures, accepted by 2024ICASSP

**SubmitDate**: 2024-01-13    [abs](http://arxiv.org/abs/2401.02727v2) [paper-pdf](http://arxiv.org/pdf/2401.02727v2)

**Authors**: Hui Zeng, Biwei Chen, Anjie Peng

**Abstract**: Adversarial examples (AEs) have been extensively studied due to their potential for privacy protection and inspiring robust neural networks. Yet, making a targeted AE transferable across unknown models remains challenging. In this paper, to alleviate the overfitting dilemma common in an AE crafted by existing simple iterative attacks, we propose fine-tuning it in the feature space. Specifically, starting with an AE generated by a baseline attack, we encourage the features conducive to the target class and discourage the features to the original class in a middle layer of the source model. Extensive experiments demonstrate that only a few iterations of fine-tuning can boost existing attacks' targeted transferability nontrivially and universally. Our results also verify that the simple iterative attacks can yield comparable or even better transferability than the resource-intensive methods, which rest on training target-specific classifiers or generators with additional data. The code is available at: github.com/zengh5/TA_feature_FT.



## **46. How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**

cs.CL

14 pages of the main text, qualitative examples of jailbreaks may be  harmful in nature

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.06373v1) [paper-pdf](http://arxiv.org/pdf/2401.06373v1)

**Authors**: Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi

**Abstract**: Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs



## **47. FlashSyn: Flash Loan Attack Synthesis via Counter Example Driven Approximation**

cs.PL

29 pages, 8 figures, conference paper extended version

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2206.10708v3) [paper-pdf](http://arxiv.org/pdf/2206.10708v3)

**Authors**: Zhiyang Chen, Sidi Mohamed Beillahi, Fan Long

**Abstract**: In decentralized finance (DeFi), lenders can offer flash loans to borrowers, i.e., loans that are only valid within a blockchain transaction and must be repaid with fees by the end of that transaction. Unlike normal loans, flash loans allow borrowers to borrow large assets without upfront collaterals deposits. Malicious adversaries use flash loans to gather large assets to exploit vulnerable DeFi protocols. In this paper, we introduce a new framework for automated synthesis of adversarial transactions that exploit DeFi protocols using flash loans. To bypass the complexity of a DeFi protocol, we propose a new technique to approximate the DeFi protocol functional behaviors using numerical methods (polynomial linear regression and nearest-neighbor interpolation). We then construct an optimization query using the approximated functions of the DeFi protocol to find an adversarial attack constituted of a sequence of functions invocations with optimal parameters that gives the maximum profit. To improve the accuracy of the approximation, we propose a novel counterexample driven approximation refinement technique. We implement our framework in a tool named FlashSyn. We evaluate FlashSyn on 16 DeFi protocols that were victims to flash loan attacks and 2 DeFi protocols from Damn Vulnerable DeFi challenges. FlashSyn automatically synthesizes an adversarial attack for 16 of the 18 benchmarks. Among the 16 successful cases, FlashSyn identifies attack vectors yielding higher profits than those employed by historical hackers in 3 cases, and also discovers multiple distinct attack vectors in 10 cases, demonstrating its effectiveness in finding possible flash loan attacks.



## **48. When Fairness Meets Privacy: Exploring Privacy Threats in Fair Binary Classifiers through Membership Inference Attacks**

cs.LG

Under review

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2311.03865v2) [paper-pdf](http://arxiv.org/pdf/2311.03865v2)

**Authors**: Huan Tian, Guangsheng Zhang, Bo Liu, Tianqing Zhu, Ming Ding, Wanlei Zhou

**Abstract**: Previous studies have developed fairness methods for biased models that exhibit discriminatory behaviors towards specific subgroups. While these models have shown promise in achieving fair predictions, recent research has identified their potential vulnerability to score-based membership inference attacks (MIAs). In these attacks, adversaries can infer whether a particular data sample was used during training by analyzing the model's prediction scores. However, our investigations reveal that these score-based MIAs are ineffective when targeting fairness-enhanced models in binary classifications. The attack models trained to launch the MIAs degrade into simplistic threshold models, resulting in lower attack performance. Meanwhile, we observe that fairness methods often lead to prediction performance degradation for the majority subgroups of the training data. This raises the barrier to successful attacks and widens the prediction gaps between member and non-member data. Building upon these insights, we propose an efficient MIA method against fairness-enhanced models based on fairness discrepancy results (FD-MIA). It leverages the difference in the predictions from both the original and fairness-enhanced models and exploits the observed prediction gaps as attack clues. We also explore potential strategies for mitigating privacy leakages. Extensive experiments validate our findings and demonstrate the efficacy of the proposed method.



## **49. MVPatch: More Vivid Patch for Adversarial Camouflaged Attacks on Object Detectors in the Physical World**

cs.CR

14 pages, 8 figures. This work has been submitted to the IEEE for  possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2312.17431v2) [paper-pdf](http://arxiv.org/pdf/2312.17431v2)

**Authors**: Zheng Zhou, Hongbo Zhao, Ju Liu, Qiaosheng Zhang, Liwei Geng, Shuchang Lyu, Wenquan Feng

**Abstract**: Recent investigations demonstrate that adversarial patches can be utilized to manipulate the result of object detection models. However, the conspicuous patterns on these patches may draw more attention and raise suspicions among humans. Moreover, existing works have primarily focused on enhancing the efficacy of attacks in the physical domain, rather than seeking to optimize their stealth attributes and transferability potential. To address these issues, we introduce a dual-perception-based attack framework that generates an adversarial patch known as the More Vivid Patch (MVPatch). The framework consists of a model-perception degradation method and a human-perception improvement method. To derive the MVPatch, we formulate an iterative process that simultaneously constrains the efficacy of multiple object detectors and refines the visual correlation between the generated adversarial patch and a realistic image. Our method employs a model-perception-based approach that reduces the object confidence scores of several object detectors to boost the transferability of adversarial patches. Further, within the human-perception-based framework, we put forward a lightweight technique for visual similarity measurement that facilitates the development of inconspicuous and natural adversarial patches and eliminates the reliance on additional generative models. Additionally, we introduce the naturalness score and transferability score as metrics for an unbiased assessment of various adversarial patches' natural appearance and transferability capacity. Extensive experiments demonstrate that the proposed MVPatch algorithm achieves superior attack transferability compared to similar algorithms in both digital and physical domains while also exhibiting a more natural appearance. These findings emphasize the remarkable stealthiness and transferability of the proposed MVPatch attack algorithm.



## **50. On the Query Complexity of Training Data Reconstruction in Private Learning**

cs.LG

Updated proof of Thm 10, fixed typos

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2303.16372v6) [paper-pdf](http://arxiv.org/pdf/2303.16372v6)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We analyze the number of queries that a whitebox adversary needs to make to a private learner in order to reconstruct its training data. For $(\epsilon, \delta)$ DP learners with training data drawn from any arbitrary compact metric space, we provide the \emph{first known lower bounds on the adversary's query complexity} as a function of the learner's privacy parameters. \emph{Our results are minimax optimal for every $\epsilon \geq 0, \delta \in [0, 1]$, covering both $\epsilon$-DP and $(0, \delta)$ DP as corollaries}. Beyond this, we obtain query complexity lower bounds for $(\alpha, \epsilon)$ R\'enyi DP learners that are valid for any $\alpha > 1, \epsilon \geq 0$. Finally, we analyze data reconstruction attacks on locally compact metric spaces via the framework of Metric DP, a generalization of DP that accounts for the underlying metric structure of the data. In this setting, we provide the first known analysis of data reconstruction in unbounded, high dimensional spaces and obtain query complexity lower bounds that are nearly tight modulo logarithmic factors.



