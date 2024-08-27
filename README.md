# Latest Adversarial Attack Papers
**update at 2024-08-27 18:58:17**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks**

cs.LG

Accepted at ECAI - 27TH EUROPEAN CONFERENCE ON ARTIFICIAL  INTELLIGENCE

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2405.12295v3) [paper-pdf](http://arxiv.org/pdf/2405.12295v3)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pregowska And Tomasz Michalak

**Abstract**: Graph Neural Networks (GNNs) are recognized as potent tools for processing real-world data organized in graph structures. Especially inductive GNNs, which allow for the processing of graph-structured data without relying on predefined graph structures, are becoming increasingly important in a wide range of applications. As such these networks become attractive targets for model-stealing attacks where an adversary seeks to replicate the functionality of the targeted network. Significant efforts have been devoted to developing model-stealing attacks that extract models trained on images and texts. However, little attention has been given to stealing GNNs trained on graph data. This paper identifies a new method of performing unsupervised model-stealing attacks against inductive GNNs, utilizing graph contrastive learning and spectral graph augmentations to efficiently extract information from the targeted model. The new type of attack is thoroughly evaluated on six datasets and the results show that our approach outperforms the current state-of-the-art by Shen et al. (2021). In particular, our attack surpasses the baseline across all benchmarks, attaining superior fidelity and downstream accuracy of the stolen model while necessitating fewer queries directed toward the target model.



## **2. Celtibero: Robust Layered Aggregation for Federated Learning**

cs.CR

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14240v1) [paper-pdf](http://arxiv.org/pdf/2408.14240v1)

**Authors**: Borja Molina-Coronado

**Abstract**: Federated Learning (FL) is an innovative approach to distributed machine learning. While FL offers significant privacy advantages, it also faces security challenges, particularly from poisoning attacks where adversaries deliberately manipulate local model updates to degrade model performance or introduce hidden backdoors. Existing defenses against these attacks have been shown to be effective when the data on the nodes is identically and independently distributed (i.i.d.), but they often fail under less restrictive, non-i.i.d data conditions. To overcome these limitations, we introduce Celtibero, a novel defense mechanism that integrates layered aggregation to enhance robustness against adversarial manipulation. Through extensive experiments on the MNIST and IMDB datasets, we demonstrate that Celtibero consistently achieves high main task accuracy (MTA) while maintaining minimal attack success rates (ASR) across a range of untargeted and targeted poisoning attacks. Our results highlight the superiority of Celtibero over existing defenses such as FL-Defender, LFighter, and FLAME, establishing it as a highly effective solution for securing federated learning systems against sophisticated poisoning attacks.



## **3. SNNGX: Securing Spiking Neural Networks with Genetic XOR Encryption on RRAM-based Neuromorphic Accelerator**

cs.CR

International Conference on Computer-Aided Design 2024

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2407.15152v2) [paper-pdf](http://arxiv.org/pdf/2407.15152v2)

**Authors**: Kwunhang Wong, Songqi Wang, Wei Huang, Xinyuan Zhang, Yangu He, Karl M. H. Lai, Yuzhong Jiao, Ning Lin, Xiaojuan Qi, Xiaoming Chen, Zhongrui Wang

**Abstract**: Biologically plausible Spiking Neural Networks (SNNs), characterized by spike sparsity, are growing tremendous attention over intellectual edge devices and critical bio-medical applications as compared to artificial neural networks (ANNs). However, there is a considerable risk from malicious attempts to extract white-box information (i.e., weights) from SNNs, as attackers could exploit well-trained SNNs for profit and white-box adversarial concerns. There is a dire need for intellectual property (IP) protective measures. In this paper, we present a novel secure software-hardware co-designed RRAM-based neuromorphic accelerator for protecting the IP of SNNs. Software-wise, we design a tailored genetic algorithm with classic XOR encryption to target the least number of weights that need encryption. From a hardware perspective, we develop a low-energy decryption module, meticulously designed to provide zero decryption latency. Extensive results from various datasets, including NMNIST, DVSGesture, EEGMMIDB, Braille Letter, and SHD, demonstrate that our proposed method effectively secures SNNs by encrypting a minimal fraction of stealthy weights, only 0.00005% to 0.016% weight bits. Additionally, it achieves a substantial reduction in energy consumption, ranging from x59 to x6780, and significantly lowers decryption latency, ranging from x175 to x4250. Moreover, our method requires as little as one sample per class in dataset for encryption and addresses hessian/gradient-based search insensitive problems. This strategy offers a highly efficient and flexible solution for securing SNNs in diverse applications.



## **4. 2D-Malafide: Adversarial Attacks Against Face Deepfake Detection Systems**

cs.CV

Accepted at BIOSIG 2024

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14143v1) [paper-pdf](http://arxiv.org/pdf/2408.14143v1)

**Authors**: Chiara Galdi, Michele Panariello, Massimiliano Todisco, Nicholas Evans

**Abstract**: We introduce 2D-Malafide, a novel and lightweight adversarial attack designed to deceive face deepfake detection systems. Building upon the concept of 1D convolutional perturbations explored in the speech domain, our method leverages 2D convolutional filters to craft perturbations which significantly degrade the performance of state-of-the-art face deepfake detectors. Unlike traditional additive noise approaches, 2D-Malafide optimises a small number of filter coefficients to generate robust adversarial perturbations which are transferable across different face images. Experiments, conducted using the FaceForensics++ dataset, demonstrate that 2D-Malafide substantially degrades detection performance in both white-box and black-box settings, with larger filter sizes having the greatest impact. Additionally, we report an explainability analysis using GradCAM which illustrates how 2D-Malafide misleads detection systems by altering the image areas used most for classification. Our findings highlight the vulnerability of current deepfake detection systems to convolutional adversarial attacks as well as the need for future work to enhance detection robustness through improved image fidelity constraints.



## **5. A Unified Membership Inference Method for Visual Self-supervised Encoder via Part-aware Capability**

cs.CV

Accepted by ACM CCS2024, Full version

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2404.02462v2) [paper-pdf](http://arxiv.org/pdf/2404.02462v2)

**Authors**: Jie Zhu, Jirong Zha, Ding Li, Leye Wang

**Abstract**: Self-supervised learning shows promise in harnessing extensive unlabeled data, but it also confronts significant privacy concerns, especially in vision. In this paper, we aim to perform membership inference on visual self-supervised models in a more realistic setting: self-supervised training method and details are unknown for an adversary when attacking as he usually faces a black-box system in practice. In this setting, considering that self-supervised model could be trained by completely different self-supervised paradigms, e.g., masked image modeling and contrastive learning, with complex training details, we propose a unified membership inference method called PartCrop. It is motivated by the shared part-aware capability among models and stronger part response on the training data. Specifically, PartCrop crops parts of objects in an image to query responses with the image in representation space. We conduct extensive attacks on self-supervised models with different training protocols and structures using three widely used image datasets. The results verify the effectiveness and generalization of PartCrop. Moreover, to defend against PartCrop, we evaluate two common approaches, i.e., early stop and differential privacy, and propose a tailored method called shrinking crop scale range. The defense experiments indicate that all of them are effective. Our code is available at https://github.com/JiePKU/PartCrop.



## **6. TF-Attack: Transferable and Fast Adversarial Attacks on Large Language Models**

cs.CL

14 pages, 6 figures. arXiv admin note: text overlap with  arXiv:2305.17440 by other authors

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.13985v1) [paper-pdf](http://arxiv.org/pdf/2408.13985v1)

**Authors**: Zelin Li, Kehai Chen, Xuefeng Bai, Lemao Liu, Mingming Yang, Yang Xiang, Min Zhang

**Abstract**: With the great advancements in large language models (LLMs), adversarial attacks against LLMs have recently attracted increasing attention. We found that pre-existing adversarial attack methodologies exhibit limited transferability and are notably inefficient, particularly when applied to LLMs. In this paper, we analyze the core mechanisms of previous predominant adversarial attack methods, revealing that 1) the distributions of importance score differ markedly among victim models, restricting the transferability; 2) the sequential attack processes induces substantial time overheads. Based on the above two insights, we introduce a new scheme, named TF-Attack, for Transferable and Fast adversarial attacks on LLMs. TF-Attack employs an external LLM as a third-party overseer rather than the victim model to identify critical units within sentences. Moreover, TF-Attack introduces the concept of Importance Level, which allows for parallel substitutions of attacks. We conduct extensive experiments on 6 widely adopted benchmarks, evaluating the proposed method through both automatic and human metrics. Results show that our method consistently surpasses previous methods in transferability and delivers significant speed improvements, up to 20 times faster than earlier attack strategies.



## **7. RT-Attack: Jailbreaking Text-to-Image Models via Random Token**

cs.CV

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2408.13896v1) [paper-pdf](http://arxiv.org/pdf/2408.13896v1)

**Authors**: Sensen Gao, Xiaojun Jia, Yihao Huang, Ranjie Duan, Jindong Gu, Yang Liu, Qing Guo

**Abstract**: Recently, Text-to-Image(T2I) models have achieved remarkable success in image generation and editing, yet these models still have many potential issues, particularly in generating inappropriate or Not-Safe-For-Work(NSFW) content. Strengthening attacks and uncovering such vulnerabilities can advance the development of reliable and practical T2I models. Most of the previous works treat T2I models as white-box systems, using gradient optimization to generate adversarial prompts. However, accessing the model's gradient is often impossible in real-world scenarios. Moreover, existing defense methods, those using gradient masking, are designed to prevent attackers from obtaining accurate gradient information. While some black-box jailbreak attacks have been explored, these typically rely on simply replacing sensitive words, leading to suboptimal attack performance. To address this issue, we introduce a two-stage query-based black-box attack method utilizing random search. In the first stage, we establish a preliminary prompt by maximizing the semantic similarity between the adversarial and target harmful prompts. In the second stage, we use this initial prompt to refine our approach, creating a detailed adversarial prompt aimed at jailbreaking and maximizing the similarity in image features between the images generated from this prompt and those produced by the target harmful prompt. Extensive experiments validate the effectiveness of our method in attacking the latest prompt checkers, post-hoc image checkers, securely trained T2I models, and online commercial models.



## **8. Hiding Backdoors within Event Sequence Data via Poisoning Attacks**

cs.LG

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2308.10201v2) [paper-pdf](http://arxiv.org/pdf/2308.10201v2)

**Authors**: Alina Ermilova, Elizaveta Kovtun, Dmitry Berestnev, Alexey Zaytsev

**Abstract**: The financial industry relies on deep learning models for making important decisions. This adoption brings new danger, as deep black-box models are known to be vulnerable to adversarial attacks. In computer vision, one can shape the output during inference by performing an adversarial attack called poisoning via introducing a backdoor into the model during training. For sequences of financial transactions of a customer, insertion of a backdoor is harder to perform, as models operate over a more complex discrete space of sequences, and systematic checks for insecurities occur. We provide a method to introduce concealed backdoors, creating vulnerabilities without altering their functionality for uncontaminated data. To achieve this, we replace a clean model with a poisoned one that is aware of the availability of a backdoor and utilize this knowledge. Our most difficult for uncovering attacks include either additional supervised detection step of poisoned data activated during the test or well-hidden model weight modifications. The experimental study provides insights into how these effects vary across different datasets, architectures, and model components. Alternative methods and baselines, such as distillation-type regularization, are also explored but found to be less efficient. Conducted on three open transaction datasets and architectures, including LSTM, CNN, and Transformer, our findings not only illuminate the vulnerabilities in contemporary models but also can drive the construction of more robust systems.



## **9. Sample-Independent Federated Learning Backdoor Attack**

cs.CR

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2408.13849v1) [paper-pdf](http://arxiv.org/pdf/2408.13849v1)

**Authors**: Weida Xu, Yang Xu, Sicong Zhang

**Abstract**: In federated learning, backdoor attacks embed triggers in the adversarial client's data to inject a backdoor into the model. To evade detection through sample analysis, non-sample-modifying backdoor attack methods based on dropout have been developed. However, these methods struggle to covertly utilize dropout in evaluation mode, thus hindering their deployment in real-world scenarios. To address these, this paper introduces GhostB, a novel approach to federated learning backdoor attacks that neither alters samples nor relies on dropout. This method employs the behavior of neurons producing specific values as triggers. By mapping these neuronal values to categories specified by the adversary, the backdoor is implanted and activated when particular feature values are detected at designated neurons. Our experiments conducted on TIMIT, LibriSpeech, and VoxCeleb2 databases in both Closed Set Identification (CSI) and Open Set Identification (OSI) scenarios demonstrate that GhostB achieves a 100% success rate upon activation, with this rate maintained across experiments involving 1 to 50 ghost neurons. This paper investigates how the dispersion of neurons and their depth within hidden layers affect the success rate, revealing that increased dispersion and positioning of neurons can significantly decrease effectiveness, potentially rendering the attack unsuccessful.



## **10. On the Robustness of Kolmogorov-Arnold Networks: An Adversarial Perspective**

cs.CV

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2408.13809v1) [paper-pdf](http://arxiv.org/pdf/2408.13809v1)

**Authors**: Tal Alter, Raz Lapid, Moshe Sipper

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently emerged as a novel approach to function approximation, demonstrating remarkable potential in various domains. Despite their theoretical promise, the robustness of KANs under adversarial conditions has yet to be thoroughly examined. In this paper, we explore the adversarial robustness of KANs, with a particular focus on image classification tasks. We assess the performance of KANs against standard white-box adversarial attacks, comparing their resilience to that of established neural network architectures. Further, we investigate the transferability of adversarial examples between KANs and Multilayer Perceptron (MLPs), deriving critical insights into the unique vulnerabilities of KANs. Our experiments use the MNIST, FashionMNIST, and KMNIST datasets, providing a comprehensive evaluation of KANs in adversarial scenarios. This work offers the first in-depth analysis of security in KANs, laying the groundwork for future research in this emerging field.



## **11. CAMH: Advancing Model Hijacking Attack in Machine Learning**

cs.CR

9 pages

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2408.13741v1) [paper-pdf](http://arxiv.org/pdf/2408.13741v1)

**Authors**: Xing He, Jiahao Chen, Yuwen Pu, Qingming Li, Chunyi Zhou, Yingcai Wu, Jinbao Li, Shouling Ji

**Abstract**: In the burgeoning domain of machine learning, the reliance on third-party services for model training and the adoption of pre-trained models have surged. However, this reliance introduces vulnerabilities to model hijacking attacks, where adversaries manipulate models to perform unintended tasks, leading to significant security and ethical concerns, like turning an ordinary image classifier into a tool for detecting faces in pornographic content, all without the model owner's knowledge. This paper introduces Category-Agnostic Model Hijacking (CAMH), a novel model hijacking attack method capable of addressing the challenges of class number mismatch, data distribution divergence, and performance balance between the original and hijacking tasks. CAMH incorporates synchronized training layers, random noise optimization, and a dual-loop optimization approach to ensure minimal impact on the original task's performance while effectively executing the hijacking task. We evaluate CAMH across multiple benchmark datasets and network architectures, demonstrating its potent attack effectiveness while ensuring minimal degradation in the performance of the original task.



## **12. Attack on Scene Flow using Point Clouds**

cs.CV

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2404.13621v4) [paper-pdf](http://arxiv.org/pdf/2404.13621v4)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. The robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants shows a higher vulnerability for the optical flow networks. Code is available at https://github.com/aheldis/Attack-on-Scene-Flow-using-Point-Clouds.git.



## **13. Interpretable and Robust AI in EEG Systems: A Survey**

eess.SP

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2304.10755v3) [paper-pdf](http://arxiv.org/pdf/2304.10755v3)

**Authors**: Xinliang Zhou, Chenyu Liu, Zhongruo Wang, Liming Zhai, Ziyu Jia, Cuntai Guan, Yang Liu

**Abstract**: The close coupling of artificial intelligence (AI) and electroencephalography (EEG) has substantially advanced human-computer interaction (HCI) technologies in the AI era. Different from traditional EEG systems, the interpretability and robustness of AI-based EEG systems are becoming particularly crucial. The interpretability clarifies the inner working mechanisms of AI models and thus can gain the trust of users. The robustness reflects the AI's reliability against attacks and perturbations, which is essential for sensitive and fragile EEG signals. Thus the interpretability and robustness of AI in EEG systems have attracted increasing attention, and their research has achieved great progress recently. However, there is still no survey covering recent advances in this field. In this paper, we present the first comprehensive survey and summarize the interpretable and robust AI techniques for EEG systems. Specifically, we first propose a taxonomy of interpretability by characterizing it into three types: backpropagation, perturbation, and inherently interpretable methods. Then we classify the robustness mechanisms into four classes: noise and artifacts, human variability, data acquisition instability, and adversarial attacks. Finally, we identify several critical and unresolved challenges for interpretable and robust AI in EEG systems and further discuss their future directions.



## **14. Shortcuts Everywhere and Nowhere: Exploring Multi-Trigger Backdoor Attacks**

cs.LG

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2401.15295v2) [paper-pdf](http://arxiv.org/pdf/2401.15295v2)

**Authors**: Yige Li, Jiabo He, Hanxun Huang, Jun Sun, Xingjun Ma

**Abstract**: Backdoor attacks have become a significant threat to the pre-training and deployment of deep neural networks (DNNs). Although numerous methods for detecting and mitigating backdoor attacks have been proposed, most rely on identifying and eliminating the ``shortcut" created by the backdoor, which links a specific source class to a target class. However, these approaches can be easily circumvented by designing multiple backdoor triggers that create shortcuts everywhere and therefore nowhere specific. In this study, we explore the concept of Multi-Trigger Backdoor Attacks (MTBAs), where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks including \textit{parallel}, \textit{sequential}, and \textit{hybrid} attacks, we demonstrate that 1) multiple triggers can coexist, overwrite, or cross-activate one another, and 2) MTBAs easily break the prevalent shortcut assumption underlying most existing backdoor detection/removal methods, rendering them ineffective. Given the security risk posed by MTBAs, we have created a multi-trigger backdoor poisoning dataset to facilitate future research on detecting and mitigating these attacks, and we also discuss potential defense strategies against MTBAs.



## **15. Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures**

cs.CV

ECCV 2024

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2403.14772v2) [paper-pdf](http://arxiv.org/pdf/2403.14772v2)

**Authors**: Sayanton V. Dibbo, Adam Breuer, Juston Moore, Michael Teti

**Abstract**: Recent model inversion attack algorithms permit adversaries to reconstruct a neural network's private and potentially sensitive training data by repeatedly querying the network. In this work, we develop a novel network architecture that leverages sparse-coding layers to obtain superior robustness to this class of attacks. Three decades of computer science research has studied sparse coding in the context of image denoising, object recognition, and adversarial misclassification settings, but to the best of our knowledge, its connection to state-of-the-art privacy vulnerabilities remains unstudied. In this work, we hypothesize that sparse coding architectures suggest an advantageous means to defend against model inversion attacks because they allow us to control the amount of irrelevant private information encoded by a network in a manner that is known to have little effect on classification accuracy. Specifically, compared to networks trained with a variety of state-of-the-art defenses, our sparse-coding architectures maintain comparable or higher classification accuracy while degrading state-of-the-art training data reconstructions by factors of 1.1 to 18.3 across a variety of reconstruction quality metrics (PSNR, SSIM, FID). This performance advantage holds across 5 datasets ranging from CelebA faces to medical images and CIFAR-10, and across various state-of-the-art SGD-based and GAN-based inversion attacks, including Plug-&-Play attacks. We provide a cluster-ready PyTorch codebase to promote research and standardize defense evaluations.



## **16. Detecting Adversarial Data via Perturbation Forgery**

cs.CV

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2405.16226v2) [paper-pdf](http://arxiv.org/pdf/2405.16226v2)

**Authors**: Qian Wang, Chen Li, Yuchen Luo, Hefei Ling, Ping Li, Jiazhong Chen, Shijuan Huang, Ning Yu

**Abstract**: As a defense strategy against adversarial attacks, adversarial detection aims to identify and filter out adversarial data from the data flow based on discrepancies in distribution and noise patterns between natural and adversarial data. Although previous detection methods achieve high performance in detecting gradient-based adversarial attacks, new attacks based on generative models with imbalanced and anisotropic noise patterns evade detection. Even worse, existing techniques either necessitate access to attack data before deploying a defense or incur a significant time cost for inference, rendering them impractical for defending against newly emerging attacks that are unseen by defenders. In this paper, we explore the proximity relationship between adversarial noise distributions and demonstrate the existence of an open covering for them. By learning to distinguish this open covering from the distribution of natural data, we can develop a detector with strong generalization capabilities against all types of adversarial attacks. Based on this insight, we heuristically propose Perturbation Forgery, which includes noise distribution perturbation, sparse mask generation, and pseudo-adversarial data production, to train an adversarial detector capable of detecting unseen gradient-based, generative-model-based, and physical adversarial attacks, while remaining agnostic to any specific models. Comprehensive experiments conducted on multiple general and facial datasets, with a wide spectrum of attacks, validate the strong generalization of our method.



## **17. Continual Adversarial Defense**

cs.CV

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2312.09481v3) [paper-pdf](http://arxiv.org/pdf/2312.09481v3)

**Authors**: Qian Wang, Yaoyao Liu, Hefei Ling, Yingwei Li, Qihao Liu, Ping Li, Jiazhong Chen, Alan Yuille, Ning Yu

**Abstract**: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. A well-matched approach to the dynamic environment lies in a defense system that continuously collects adversarial data online to quickly improve itself. Therefore, we put forward a practical defense deployment against a challenging threat model and propose, for the first time, the Continual Adversarial Defense (CAD) framework that adapts to attack sequences under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accuracy on both clean and adversarial data. We explore and integrate cutting-edge continual learning, few-shot learning, and ensemble learning techniques to qualify the principles. Extensive experiments validate the effectiveness of our approach against multiple stages of modern adversarial attacks and demonstrate significant improvements over numerous baseline methods. In particular, CAD is capable of quickly adapting with minimal budget and a low cost of defense failure while maintaining good performance against previous attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.



## **18. Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors**

cs.CV

15 pages

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2405.10529v2) [paper-pdf](http://arxiv.org/pdf/2405.10529v2)

**Authors**: Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, Chaowei Xiao

**Abstract**: Large language models have become increasingly prominent, also signaling a shift towards multimodality as the next frontier in artificial intelligence, where their embeddings are harnessed as prompts to generate textual content. Vision-language models (VLMs) stand at the forefront of this advancement, offering innovative ways to combine visual and textual data for enhanced understanding and interaction. However, this integration also enlarges the attack surface. Patch-based adversarial attack is considered the most realistic threat model in physical vision applications, as demonstrated in many existing literature. In this paper, we propose to address patched visual prompt injection, where adversaries exploit adversarial patches to generate target content in VLMs. Our investigation reveals that patched adversarial prompts exhibit sensitivity to pixel-wise randomization, a trait that remains robust even against adaptive attacks designed to counteract such defenses. Leveraging this insight, we introduce SmoothVLM, a defense mechanism rooted in smoothing techniques, specifically tailored to protect VLMs from the threat of patched visual prompt injectors. Our framework significantly lowers the attack success rate to a range between 0% and 5.0% on two leading VLMs, while achieving around 67.3% to 95.0% context recovery of the benign images, demonstrating a balance between security and usability.



## **19. Probing the Robustness of Vision-Language Pretrained Models: A Multimodal Adversarial Attack Approach**

cs.CV

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2408.13461v1) [paper-pdf](http://arxiv.org/pdf/2408.13461v1)

**Authors**: Jiwei Guan, Tianyu Ding, Longbing Cao, Lei Pan, Chen Wang, Xi Zheng

**Abstract**: Vision-language pretraining (VLP) with transformers has demonstrated exceptional performance across numerous multimodal tasks. However, the adversarial robustness of these models has not been thoroughly investigated. Existing multimodal attack methods have largely overlooked cross-modal interactions between visual and textual modalities, particularly in the context of cross-attention mechanisms. In this paper, we study the adversarial vulnerability of recent VLP transformers and design a novel Joint Multimodal Transformer Feature Attack (JMTFA) that concurrently introduces adversarial perturbations in both visual and textual modalities under white-box settings. JMTFA strategically targets attention relevance scores to disrupt important features within each modality, generating adversarial samples by fusing perturbations and leading to erroneous model predictions. Experimental results indicate that the proposed approach achieves high attack success rates on vision-language understanding and reasoning downstream tasks compared to existing baselines. Notably, our findings reveal that the textual modality significantly influences the complex fusion processes within VLP transformers. Moreover, we observe no apparent relationship between model size and adversarial robustness under our proposed attacks. These insights emphasize a new dimension of adversarial robustness and underscore potential risks in the reliable deployment of multimodal AI systems.



## **20. Toward Improving Synthetic Audio Spoofing Detection Robustness via Meta-Learning and Disentangled Training With Adversarial Examples**

cs.SD

IEEE ACCESS 2024

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.13341v1) [paper-pdf](http://arxiv.org/pdf/2408.13341v1)

**Authors**: Zhenyu Wang, John H. L. Hansen

**Abstract**: Advances in automatic speaker verification (ASV) promote research into the formulation of spoofing detection systems for real-world applications. The performance of ASV systems can be degraded severely by multiple types of spoofing attacks, namely, synthetic speech (SS), voice conversion (VC), replay, twins and impersonation, especially in the case of unseen synthetic spoofing attacks. A reliable and robust spoofing detection system can act as a security gate to filter out spoofing attacks instead of having them reach the ASV system. A weighted additive angular margin loss is proposed to address the data imbalance issue, and different margins has been assigned to improve generalization to unseen spoofing attacks in this study. Meanwhile, we incorporate a meta-learning loss function to optimize differences between the embeddings of support versus query set in order to learn a spoofing-category-independent embedding space for utterances. Furthermore, we craft adversarial examples by adding imperceptible perturbations to spoofing speech as a data augmentation strategy, then we use an auxiliary batch normalization (BN) to guarantee that corresponding normalization statistics are performed exclusively on the adversarial examples. Additionally, A simple attention module is integrated into the residual block to refine the feature extraction process. Evaluation results on the Logical Access (LA) track of the ASVspoof 2019 corpus provides confirmation of our proposed approaches' effectiveness in terms of a pooled EER of 0.87%, and a min t-DCF of 0.0277. These advancements offer effective options to reduce the impact of spoofing attacks on voice recognition/authentication systems.



## **21. Dynamic Label Adversarial Training for Deep Learning Robustness Against Adversarial Attacks**

cs.LG

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.13102v1) [paper-pdf](http://arxiv.org/pdf/2408.13102v1)

**Authors**: Zhenyu Liu, Haoran Duan, Huizhi Liang, Yang Long, Vaclav Snasel, Guiseppe Nicosia, Rajiv Ranjan, Varun Ojha

**Abstract**: Adversarial training is one of the most effective methods for enhancing model robustness. Recent approaches incorporate adversarial distillation in adversarial training architectures. However, we notice two scenarios of defense methods that limit their performance: (1) Previous methods primarily use static ground truth for adversarial training, but this often causes robust overfitting; (2) The loss functions are either Mean Squared Error or KL-divergence leading to a sub-optimal performance on clean accuracy. To solve those problems, we propose a dynamic label adversarial training (DYNAT) algorithm that enables the target model to gradually and dynamically gain robustness from the guide model's decisions. Additionally, we found that a budgeted dimension of inner optimization for the target model may contribute to the trade-off between clean accuracy and robust accuracy. Therefore, we propose a novel inner optimization method to be incorporated into the adversarial training. This will enable the target model to adaptively search for adversarial examples based on dynamic labels from the guiding model, contributing to the robustness of the target model. Extensive experiments validate the superior performance of our approach.



## **22. Robust Diffusion Models for Adversarial Purification**

cs.CV

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2403.16067v3) [paper-pdf](http://arxiv.org/pdf/2403.16067v3)

**Authors**: Guang Lin, Zerui Tao, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also mitigate the accuracy-robustness trade-off of DMs for the first time, which also provides DM-based AP an efficient adaptive ability to new attacks. Extensive experiments are conducted on CIFAR-10, CIFAR-100 and ImageNet to demonstrate that our method achieves the state-of-the-art results and exhibits generalization against different attacks.



## **23. Robust Feature Inference: A Test-time Defense Strategy using Spectral Projections**

cs.LG

Published in TMLR (28 pages, 6 figures, 20 tables)

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2307.11672v2) [paper-pdf](http://arxiv.org/pdf/2307.11672v2)

**Authors**: Anurag Singh, Mahalakshmi Sabanayagam, Krikamol Muandet, Debarghya Ghoshdastidar

**Abstract**: Test-time defenses are used to improve the robustness of deep neural networks to adversarial examples during inference. However, existing methods either require an additional trained classifier to detect and correct the adversarial samples, or perform additional complex optimization on the model parameters or the input to adapt to the adversarial samples at test-time, resulting in a significant increase in the inference time compared to the base model. In this work, we propose a novel test-time defense strategy called Robust Feature Inference (RFI) that is easy to integrate with any existing (robust) training procedure without additional test-time computation. Based on the notion of robustness of features that we present, the key idea is to project the trained models to the most robust feature space, thereby reducing the vulnerability to adversarial attacks in non-robust directions. We theoretically characterize the subspace of the eigenspectrum of the feature covariance that is the most robust for a generalized additive model. Our extensive experiments on CIFAR-10, CIFAR-100, tiny ImageNet and ImageNet datasets for several robustness benchmarks, including the state-of-the-art methods in RobustBench show that RFI improves robustness across adaptive and transfer attacks consistently. We also compare RFI with adaptive test-time defenses to demonstrate the effectiveness of our proposed approach.



## **24. Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**

cs.CV

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2401.16352v4) [paper-pdf](http://arxiv.org/pdf/2401.16352v4)

**Authors**: Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline to acquire the robust purifier model, named Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks, resulting in the robustness generalization to unseen attacks, and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves optimal robustness and exhibits generalization ability against unseen attacks.



## **25. BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models**

cs.AI

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.12798v1) [paper-pdf](http://arxiv.org/pdf/2408.12798v1)

**Authors**: Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, Jun Sun

**Abstract**: Generative Large Language Models (LLMs) have made significant strides across various tasks, but they remain vulnerable to backdoor attacks, where specific triggers in the prompt cause the LLM to generate adversary-desired responses. While most backdoor research has focused on vision or text classification tasks, backdoor attacks in text generation have been largely overlooked. In this work, we introduce \textit{BackdoorLLM}, the first comprehensive benchmark for studying backdoor attacks on LLMs. \textit{BackdoorLLM} features: 1) a repository of backdoor benchmarks with a standardized training pipeline, 2) diverse attack strategies, including data poisoning, weight poisoning, hidden state attacks, and chain-of-thought attacks, 3) extensive evaluations with over 200 experiments on 8 attacks across 7 scenarios and 6 model architectures, and 4) key insights into the effectiveness and limitations of backdoors in LLMs. We hope \textit{BackdoorLLM} will raise awareness of backdoor threats and contribute to advancing AI safety. The code is available at \url{https://github.com/bboylyg/BackdoorLLM}.



## **26. BankTweak: Adversarial Attack against Multi-Object Trackers by Manipulating Feature Banks**

cs.CV

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.12727v1) [paper-pdf](http://arxiv.org/pdf/2408.12727v1)

**Authors**: Woojin Shin, Donghwa Kang, Daejin Choi, Brent Kang, Jinkyu Lee, Hyeongboo Baek

**Abstract**: Multi-object tracking (MOT) aims to construct moving trajectories for objects, and modern multi-object trackers mainly utilize the tracking-by-detection methodology. Initial approaches to MOT attacks primarily aimed to degrade the detection quality of the frames under attack, thereby reducing accuracy only in those specific frames, highlighting a lack of \textit{efficiency}. To improve efficiency, recent advancements manipulate object positions to cause persistent identity (ID) switches during the association phase, even after the attack ends within a few frames. However, these position-manipulating attacks have inherent limitations, as they can be easily counteracted by adjusting distance-related parameters in the association phase, revealing a lack of \textit{robustness}. In this paper, we present \textsf{BankTweak}, a novel adversarial attack designed for MOT trackers, which features efficiency and robustness. \textsf{BankTweak} focuses on the feature extractor in the association phase and reveals vulnerability in the Hungarian matching method used by feature-based MOT systems. Exploiting the vulnerability, \textsf{BankTweak} induces persistent ID switches (addressing \textit{efficiency}) even after the attack ends by strategically injecting altered features into the feature banks without modifying object positions (addressing \textit{robustness}). To demonstrate the applicability, we apply \textsf{BankTweak} to three multi-object trackers (DeepSORT, StrongSORT, and MOTDT) with one-stage, two-stage, anchor-free, and transformer detectors. Extensive experiments on the MOT17 and MOT20 datasets show that our method substantially surpasses existing attacks, exposing the vulnerability of the tracking-by-detection framework to \textsf{BankTweak}.



## **27. Enhancing Transferability of Adversarial Attacks with GE-AdvGAN+: A Comprehensive Framework for Gradient Editing**

cs.AI

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.12673v1) [paper-pdf](http://arxiv.org/pdf/2408.12673v1)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Yuchen Zhang, Jiahao Huang, Jianlong Zhou, Fang Chen

**Abstract**: Transferable adversarial attacks pose significant threats to deep neural networks, particularly in black-box scenarios where internal model information is inaccessible. Studying adversarial attack methods helps advance the performance of defense mechanisms and explore model vulnerabilities. These methods can uncover and exploit weaknesses in models, promoting the development of more robust architectures. However, current methods for transferable attacks often come with substantial computational costs, limiting their deployment and application, especially in edge computing scenarios. Adversarial generative models, such as Generative Adversarial Networks (GANs), are characterized by their ability to generate samples without the need for retraining after an initial training phase. GE-AdvGAN, a recent method for transferable adversarial attacks, is based on this principle. In this paper, we propose a novel general framework for gradient editing-based transferable attacks, named GE-AdvGAN+, which integrates nearly all mainstream attack methods to enhance transferability while significantly reducing computational resource consumption. Our experiments demonstrate the compatibility and effectiveness of our framework. Compared to the baseline AdvGAN, our best-performing method, GE-AdvGAN++, achieves an average ASR improvement of 47.8. Additionally, it surpasses the latest competing algorithm, GE-AdvGAN, with an average ASR increase of 5.9. The framework also exhibits enhanced computational efficiency, achieving 2217.7 FPS, outperforming traditional methods such as BIM and MI-FGSM. The implementation code for our GE-AdvGAN+ framework is available at https://github.com/GEAdvGANP



## **28. Leveraging Information Consistency in Frequency and Spatial Domain for Adversarial Attacks**

cs.LG

Accepted by PRICAI 2024

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.12670v1) [paper-pdf](http://arxiv.org/pdf/2408.12670v1)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Yiyun Huang, Huaming Chen

**Abstract**: Adversarial examples are a key method to exploit deep neural networks. Using gradient information, such examples can be generated in an efficient way without altering the victim model. Recent frequency domain transformation has further enhanced the transferability of such adversarial examples, such as spectrum simulation attack. In this work, we investigate the effectiveness of frequency domain-based attacks, aligning with similar findings in the spatial domain. Furthermore, such consistency between the frequency and spatial domains provides insights into how gradient-based adversarial attacks induce perturbations across different domains, which is yet to be explored. Hence, we propose a simple, effective, and scalable gradient-based adversarial attack algorithm leveraging the information consistency in both frequency and spatial domains. We evaluate the algorithm for its effectiveness against different models. Extensive experiments demonstrate that our algorithm achieves state-of-the-art results compared to other gradient-based algorithms. Our code is available at: https://github.com/LMBTough/FSA.



## **29. Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks**

cs.CR

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.08924v2) [paper-pdf](http://arxiv.org/pdf/2408.08924v2)

**Authors**: Jiawei Zhao, Kejiang Chen, Xiaojian Yuan, Weiming Zhang

**Abstract**: In recent years, the rapid development of large language models (LLMs) has achieved remarkable performance across various tasks. However, research indicates that LLMs are vulnerable to jailbreak attacks, where adversaries can induce the generation of harmful content through meticulously crafted prompts. This vulnerability poses significant challenges to the secure use and promotion of LLMs. Existing defense methods offer protection from different perspectives but often suffer from insufficient effectiveness or a significant impact on the model's capabilities. In this paper, we propose a plug-and-play and easy-to-deploy jailbreak defense framework, namely Prefix Guidance (PG), which guides the model to identify harmful prompts by directly setting the first few tokens of the model's output. This approach combines the model's inherent security capabilities with an external classifier to defend against jailbreak attacks. We demonstrate the effectiveness of PG across three models and five attack methods. Compared to baselines, our approach is generally more effective on average. Additionally, results on the Just-Eval benchmark further confirm PG's superiority to preserve the model's performance. our code is available at https://github.com/weiyezhimeng/Prefix-Guidance.



## **30. Talos: A More Effective and Efficient Adversarial Defense for GNN Models Based on the Global Homophily of Graphs**

cs.LG

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2406.03833v2) [paper-pdf](http://arxiv.org/pdf/2406.03833v2)

**Authors**: Duanyu Li, Huijun Wu, Min Xie, Xugang Wu, Zhenwei Wu, Wenzhe Zhang

**Abstract**: Graph neural network (GNN) models play a pivotal role in numerous tasks involving graph-related data analysis. Despite their efficacy, similar to other deep learning models, GNNs are susceptible to adversarial attacks. Even minor perturbations in graph data can induce substantial alterations in model predictions. While existing research has explored various adversarial defense techniques for GNNs, the challenge of defending against adversarial attacks on real-world scale graph data remains largely unresolved. On one hand, methods reliant on graph purification and preprocessing tend to excessively emphasize local graph information, leading to sub-optimal defensive outcomes. On the other hand, approaches rooted in graph structure learning entail significant time overheads, rendering them impractical for large-scale graphs. In this paper, we propose a new defense method named Talos, which enhances the global, rather than local, homophily of graphs as a defense. Experiments show that the proposed approach notably outperforms state-of-the-art defense approaches, while imposing little computational overhead.



## **31. Regularization for Adversarial Robust Learning**

cs.LG

51 pages, 5 figures

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.09672v2) [paper-pdf](http://arxiv.org/pdf/2408.09672v2)

**Authors**: Jie Wang, Rui Gao, Yao Xie

**Abstract**: Despite the growing prevalence of artificial neural networks in real-world applications, their vulnerability to adversarial attacks remains a significant concern, which motivates us to investigate the robustness of machine learning models. While various heuristics aim to optimize the distributionally robust risk using the $\infty$-Wasserstein metric, such a notion of robustness frequently encounters computation intractability. To tackle the computational challenge, we develop a novel approach to adversarial training that integrates $\phi$-divergence regularization into the distributionally robust risk function. This regularization brings a notable improvement in computation compared with the original formulation. We develop stochastic gradient methods with biased oracles to solve this problem efficiently, achieving the near-optimal sample complexity. Moreover, we establish its regularization effects and demonstrate it is asymptotic equivalence to a regularized empirical risk minimization framework, by considering various scaling regimes of the regularization parameter and robustness level. These regimes yield gradient norm regularization, variance regularization, or a smoothed gradient norm regularization that interpolates between these extremes. We numerically validate our proposed method in supervised learning, reinforcement learning, and contextual learning and showcase its state-of-the-art performance against various adversarial attacks.



## **32. Adversarial Examples in the Physical World: A Survey**

cs.CV

Adversarial examples, physical-world scenarios, attacks and defenses

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2311.01473v3) [paper-pdf](http://arxiv.org/pdf/2311.01473v3)

**Authors**: Jiakai Wang, Xianglong Liu, Jin Hu, Donghua Wang, Siyang Wu, Tingsong Jiang, Yuanfang Guo, Aishan Liu, Jiantao Zhou

**Abstract**: Deep neural networks (DNNs) have demonstrated high vulnerability to adversarial examples, raising broad security concerns about their applications. Besides the attacks in the digital world, the practical implications of adversarial examples in the physical world present significant challenges and safety concerns. However, current research on physical adversarial examples (PAEs) lacks a comprehensive understanding of their unique characteristics, leading to limited significance and understanding. In this paper, we address this gap by thoroughly examining the characteristics of PAEs within a practical workflow encompassing training, manufacturing, and re-sampling processes. By analyzing the links between physical adversarial attacks, we identify manufacturing and re-sampling as the primary sources of distinct attributes and particularities in PAEs. Leveraging this knowledge, we develop a comprehensive analysis and classification framework for PAEs based on their specific characteristics, covering over 100 studies on physical-world adversarial examples. Furthermore, we investigate defense strategies against PAEs and identify open challenges and opportunities for future research. We aim to provide a fresh, thorough, and systematic understanding of PAEs, thereby promoting the development of robust adversarial learning and its application in open-world scenarios to provide the community with a continuously updated list of physical world adversarial sample resources, including papers, code, \etc, within the proposed framework



## **33. Query-Efficient Video Adversarial Attack with Stylized Logo**

cs.CV

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.12099v1) [paper-pdf](http://arxiv.org/pdf/2408.12099v1)

**Authors**: Duoxun Tang, Yuxin Cao, Xi Xiao, Derui Wang, Sheng Wen, Tianqing Zhu

**Abstract**: Video classification systems based on Deep Neural Networks (DNNs) have demonstrated excellent performance in accurately verifying video content. However, recent studies have shown that DNNs are highly vulnerable to adversarial examples. Therefore, a deep understanding of adversarial attacks can better respond to emergency situations. In order to improve attack performance, many style-transfer-based attacks and patch-based attacks have been proposed. However, the global perturbation of the former will bring unnatural global color, while the latter is difficult to achieve success in targeted attacks due to the limited perturbation space. Moreover, compared to a plethora of methods targeting image classifiers, video adversarial attacks are still not that popular. Therefore, to generate adversarial examples with a low budget and to provide them with a higher verisimilitude, we propose a novel black-box video attack framework, called Stylized Logo Attack (SLA). SLA is conducted through three steps. The first step involves building a style references set for logos, which can not only make the generated examples more natural, but also carry more target class features in the targeted attacks. Then, reinforcement learning (RL) is employed to determine the style reference and position parameters of the logo within the video, which ensures that the stylized logo is placed in the video with optimal attributes. Finally, perturbation optimization is designed to optimize perturbations to improve the fooling rate in a step-by-step manner. Sufficient experimental results indicate that, SLA can achieve better performance than state-of-the-art methods and still maintain good deception effects when facing various defense methods.



## **34. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

cs.CR

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2403.05030v4) [paper-pdf](http://arxiv.org/pdf/2403.05030v4)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without leveraging knowledge of what they are or using inputs that elicit them. LAT makes use of the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. Here, we use it to defend against failure modes without examples that elicit them. Specifically, we use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.



## **35. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2407.15549v2) [paper-pdf](http://arxiv.org/pdf/2407.15549v2)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.



## **36. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2402.06255v3) [paper-pdf](http://arxiv.org/pdf/2402.06255v3)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreak attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly with a particular focus on harmful content filtering or heuristical defensive prompt designs. However, how to achieve intrinsic robustness through the prompts remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both grey-box and black-box attacks, reducing the success rate of advanced attacks to nearly 0 while maintaining the model's utility on the benign task. The proposed defense strategy incurs only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/rain152/PAT.



## **37. Competence-Based Analysis of Language Models**

cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2303.00333v4) [paper-pdf](http://arxiv.org/pdf/2303.00333v4)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain and predict behaviors across these tasks.



## **38. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11749v1) [paper-pdf](http://arxiv.org/pdf/2408.11749v1)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.



## **39. First line of defense: A robust first layer mitigates adversarial attacks**

cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11680v1) [paper-pdf](http://arxiv.org/pdf/2408.11680v1)

**Authors**: Janani Suresh, Nancy Nayak, Sheetal Kalyani

**Abstract**: Adversarial training (AT) incurs significant computational overhead, leading to growing interest in designing inherently robust architectures. We demonstrate that a carefully designed first layer of the neural network can serve as an implicit adversarial noise filter (ANF). This filter is created using a combination of large kernel size, increased convolution filters, and a maxpool operation. We show that integrating this filter as the first layer in architectures such as ResNet, VGG, and EfficientNet results in adversarially robust networks. Our approach achieves higher adversarial accuracies than existing natively robust architectures without AT and is competitive with adversarial-trained architectures across a wide range of datasets. Supporting our findings, we show that (a) the decision regions for our method have better margins, (b) the visualized loss surfaces are smoother, (c) the modified peak signal-to-noise ratio (mPSNR) values at the output of the ANF are higher, (d) high-frequency components are more attenuated, and (e) architectures incorporating ANF exhibit better denoising in Gaussian noise compared to baseline architectures. Code for all our experiments are available at \url{https://github.com/janani-suresh-97/first-line-defence.git}.



## **40. Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection**

cs.CV

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11408v1) [paper-pdf](http://arxiv.org/pdf/2408.11408v1)

**Authors**: Jingwei Sun, Xuchong Zhang, Changfeng Sun, Qicheng Bai, Hongbin Sun

**Abstract**: Multi-View Diffusion Models (MVDMs) enable remarkable improvements in the field of 3D geometric reconstruction, but the issue regarding intellectual property has received increasing attention due to unauthorized imitation. Recently, some works have utilized adversarial attacks to protect copyright. However, all these works focus on single-image generation tasks which only need to consider the inner feature of images. Previous methods are inefficient in attacking MVDMs because they lack the consideration of disrupting the geometric and visual consistency among the generated multi-view images. This paper is the first to address the intellectual property infringement issue arising from MVDMs. Accordingly, we propose a novel latent feature and attention dual erasure attack to disrupt the distribution of latent feature and the consistency across the generated images from multi-view and multi-domain simultaneously. The experiments conducted on SOTA MVDMs indicate that our approach achieves superior performances in terms of attack effectiveness, transferability, and robustness against defense methods. Therefore, this paper provides an efficient solution to protect 3D assets from MVDMs-based 3D geometry reconstruction.



## **41. AntifakePrompt: Prompt-Tuned Vision-Language Models are Fake Image Detectors**

cs.CV

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2310.17419v3) [paper-pdf](http://arxiv.org/pdf/2310.17419v3)

**Authors**: You-Ming Chang, Chen Yeh, Wei-Chen Chiu, Ning Yu

**Abstract**: Deep generative models can create remarkably photorealistic fake images while raising concerns about misinformation and copyright infringement, known as deepfake threats. Deepfake detection technique is developed to distinguish between real and fake images, where the existing methods typically learn classifiers in the image domain or various feature domains. However, the generalizability of deepfake detection against emerging and more advanced generative models remains challenging. In this paper, being inspired by the zero-shot advantages of Vision-Language Models (VLMs), we propose a novel approach called AntifakePrompt, using VLMs (e.g., InstructBLIP) and prompt tuning techniques to improve the deepfake detection accuracy over unseen data. We formulate deepfake detection as a visual question answering problem, and tune soft prompts for InstructBLIP to answer the real/fake information of a query image. We conduct full-spectrum experiments on datasets from a diversity of 3 held-in and 20 held-out generative models, covering modern text-to-image generation, image editing and adversarial image attacks. These testing datasets provide useful benchmarks in the realm of deepfake detection for further research. Moreover, results demonstrate that (1) the deepfake detection accuracy can be significantly and consistently improved (from 71.06% to 92.11%, in average accuracy over unseen domains) using pretrained vision-language models with prompt tuning; (2) our superior performance is at less cost of training data and trainable parameters, resulting in an effective and efficient solution for deepfake detection. Code and models can be found at https://github.com/nctu-eva-lab/AntifakePrompt.



## **42. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

physics.soc-ph

17 pages, 4 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.19692v3) [paper-pdf](http://arxiv.org/pdf/2406.19692v3)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.



## **43. Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis**

cs.LG

33 pages

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2407.11463v2) [paper-pdf](http://arxiv.org/pdf/2407.11463v2)

**Authors**: Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks are a potential threat to machine learning models by causing incorrect predictions through imperceptible perturbations to the input data. While these attacks have been extensively studied in unstructured data like images, applying them to tabular data, poses new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ from the image data. To account for this distinction, it is necessary to establish tailored imperceptibility criteria specific to tabular data. However, there is currently a lack of standardised metrics for assessing the imperceptibility of adversarial attacks on tabular data. To address this gap, we propose a set of key properties and corresponding metrics designed to comprehensively characterise imperceptible adversarial attacks on tabular data. These are: proximity to the original input, sparsity of altered features, deviation from the original data distribution, sensitivity in perturbing features with narrow distribution, immutability of certain features that should remain unchanged, feasibility of specific feature values that should not go beyond valid practical ranges, and feature interdependencies capturing complex relationships between data attributes. We evaluate the imperceptibility of five adversarial attacks, including both bounded attacks and unbounded attacks, on tabular data using the proposed imperceptibility metrics. The results reveal a trade-off between the imperceptibility and effectiveness of these attacks. The study also identifies limitations in current attack algorithms, offering insights that can guide future research in the area. The findings gained from this empirical analysis provide valuable direction for enhancing the design of adversarial attack algorithms, thereby advancing adversarial machine learning on tabular data.



## **44. Unlocking Adversarial Suffix Optimization Without Affirmative Phrases: Efficient Black-box Jailbreaking via LLM as Optimizer**

cs.AI

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11313v1) [paper-pdf](http://arxiv.org/pdf/2408.11313v1)

**Authors**: Weipeng Jiang, Zhenting Wang, Juan Zhai, Shiqing Ma, Zhengyu Zhao, Chao Shen

**Abstract**: Despite prior safety alignment efforts, mainstream LLMs can still generate harmful and unethical content when subjected to jailbreaking attacks. Existing jailbreaking methods fall into two main categories: template-based and optimization-based methods. The former requires significant manual effort and domain knowledge, while the latter, exemplified by Greedy Coordinate Gradient (GCG), which seeks to maximize the likelihood of harmful LLM outputs through token-level optimization, also encounters several limitations: requiring white-box access, necessitating pre-constructed affirmative phrase, and suffering from low efficiency. In this paper, we present ECLIPSE, a novel and efficient black-box jailbreaking method utilizing optimizable suffixes. Drawing inspiration from LLMs' powerful generation and optimization capabilities, we employ task prompts to translate jailbreaking goals into natural language instructions. This guides the LLM to generate adversarial suffixes for malicious queries. In particular, a harmfulness scorer provides continuous feedback, enabling LLM self-reflection and iterative optimization to autonomously and efficiently produce effective suffixes. Experimental results demonstrate that ECLIPSE achieves an average attack success rate (ASR) of 0.92 across three open-source LLMs and GPT-3.5-Turbo, significantly surpassing GCG in 2.4 times. Moreover, ECLIPSE is on par with template-based methods in ASR while offering superior attack efficiency, reducing the average attack overhead by 83%.



## **45. EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models**

cs.AI

19 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11308v1) [paper-pdf](http://arxiv.org/pdf/2408.11308v1)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. Built upon this idea, we introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85\% in comparison with 50\% for the present SOTAs, with minimal impact on the utility and effectiveness of LLMs.



## **46. Correlation Analysis of Adversarial Attack in Time Series Classification**

cs.LG

15 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11264v1) [paper-pdf](http://arxiv.org/pdf/2408.11264v1)

**Authors**: Zhengyang Li, Wenhao Liang, Chang Dong, Weitong Chen, Dong Huang

**Abstract**: This study investigates the vulnerability of time series classification models to adversarial attacks, with a focus on how these models process local versus global information under such conditions. By leveraging the Normalized Auto Correlation Function (NACF), an exploration into the inclination of neural networks is conducted. It is demonstrated that regularization techniques, particularly those employing Fast Fourier Transform (FFT) methods and targeting frequency components of perturbations, markedly enhance the effectiveness of attacks. Meanwhile, the defense strategies, like noise introduction and Gaussian filtering, are shown to significantly lower the Attack Success Rate (ASR), with approaches based on noise introducing notably effective in countering high-frequency distortions. Furthermore, models designed to prioritize global information are revealed to possess greater resistance to adversarial manipulations. These results underline the importance of designing attack and defense mechanisms, informed by frequency domain analysis, as a means to considerably reinforce the resilience of neural network models against adversarial threats.



## **47. Revisiting Min-Max Optimization Problem in Adversarial Training**

cs.CV

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11218v1) [paper-pdf](http://arxiv.org/pdf/2408.11218v1)

**Authors**: Sina Hajer Ahmadi, Hassan Bahrami

**Abstract**: The rise of computer vision applications in the real world puts the security of the deep neural networks at risk. Recent works demonstrate that convolutional neural networks are susceptible to adversarial examples - where the input images look similar to the natural images but are classified incorrectly by the model. To provide a rebuttal to this problem, we propose a new method to build robust deep neural networks against adversarial attacks by reformulating the saddle point optimization problem in \cite{madry2017towards}. Our proposed method offers significant resistance and a concrete security guarantee against multiple adversaries. The goal of this paper is to act as a stepping stone for a new variation of deep learning models which would lead towards fully robust deep learning models.



## **48. Makeup-Guided Facial Privacy Protection via Untrained Neural Network Priors**

cs.CV

Proceedings of ECCV Workshop on Explainable AI for Biometrics, 2024

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.12387v1) [paper-pdf](http://arxiv.org/pdf/2408.12387v1)

**Authors**: Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar

**Abstract**: Deep learning-based face recognition (FR) systems pose significant privacy risks by tracking users without their consent. While adversarial attacks can protect privacy, they often produce visible artifacts compromising user experience. To mitigate this issue, recent facial privacy protection approaches advocate embedding adversarial noise into the natural looking makeup styles. However, these methods require training on large-scale makeup datasets that are not always readily available. In addition, these approaches also suffer from dataset bias. For instance, training on makeup data that predominantly contains female faces could compromise protection efficacy for male faces. To handle these issues, we propose a test-time optimization approach that solely optimizes an untrained neural network to transfer makeup style from a reference to a source image in an adversarial manner. We introduce two key modules: a correspondence module that aligns regions between reference and source images in latent space, and a decoder with conditional makeup layers. The untrained decoder, optimized via carefully designed structural and makeup consistency losses, generates a protected image that resembles the source but incorporates adversarial makeup to deceive FR models. As our approach does not rely on training with makeup face datasets, it avoids potential male/female dataset biases while providing effective protection. We further extend the proposed approach to videos by leveraging on temporal correlations. Experiments on benchmark datasets demonstrate superior performance in face verification and identification tasks and effectiveness against commercial FR systems. Our code and models will be available at https://github.com/fahadshamshad/deep-facial-privacy-prior



## **49. GAIM: Attacking Graph Neural Networks via Adversarial Influence Maximization**

cs.LG

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10948v1) [paper-pdf](http://arxiv.org/pdf/2408.10948v1)

**Authors**: Xiaodong Yang, Xiaoting Li, Huiyuan Chen, Yiwei Cai

**Abstract**: Recent studies show that well-devised perturbations on graph structures or node features can mislead trained Graph Neural Network (GNN) models. However, these methods often overlook practical assumptions, over-rely on heuristics, or separate vital attack components. In response, we present GAIM, an integrated adversarial attack method conducted on a node feature basis while considering the strict black-box setting. Specifically, we define an adversarial influence function to theoretically assess the adversarial impact of node perturbations, thereby reframing the GNN attack problem into the adversarial influence maximization problem. In our approach, we unify the selection of the target node and the construction of feature perturbations into a single optimization problem, ensuring a unique and consistent feature perturbation for each target node. We leverage a surrogate model to transform this problem into a solvable linear programming task, streamlining the optimization process. Moreover, we extend our method to accommodate label-oriented attacks, broadening its applicability. Thorough evaluations on five benchmark datasets across three popular models underscore the effectiveness of our method in both untargeted and label-oriented targeted attacks. Through comprehensive analysis and ablation studies, we demonstrate the practical value and efficacy inherent to our design choices.



## **50. A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse**

cs.CV

21 pages, 7 figures, 10 tables

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10901v1) [paper-pdf](http://arxiv.org/pdf/2408.10901v1)

**Authors**: Zhongliang Guo, Lei Fang, Jingyu Lin, Yifei Qian, Shuai Zhao, Zeyu Wang, Junhao Dong, Cunjian Chen, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Recent advancements in generative AI, particularly Latent Diffusion Models (LDMs), have revolutionized image synthesis and manipulation. However, these generative techniques raises concerns about data misappropriation and intellectual property infringement. Adversarial attacks on machine learning models have been extensively studied, and a well-established body of research has extended these techniques as a benign metric to prevent the underlying misuse of generative AI. Current approaches to safeguarding images from manipulation by LDMs are limited by their reliance on model-specific knowledge and their inability to significantly degrade semantic quality of generated images. In response to these shortcomings, we propose the Posterior Collapse Attack (PCA) based on the observation that VAEs suffer from posterior collapse during training. Our method minimizes dependence on the white-box information of target models to get rid of the implicit reliance on model-specific knowledge. By accessing merely a small amount of LDM parameters, in specific merely the VAE encoder of LDMs, our method causes a substantial semantic collapse in generation quality, particularly in perceptual consistency, and demonstrates strong transferability across various model architectures. Experimental results show that PCA achieves superior perturbation effects on image generation of LDMs with lower runtime and VRAM. Our method outperforms existing techniques, offering a more robust and generalizable solution that is helpful in alleviating the socio-technical challenges posed by the rapidly evolving landscape of generative AI.



