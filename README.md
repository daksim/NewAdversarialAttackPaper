# Latest Adversarial Attack Papers
**update at 2023-12-08 09:48:48**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Memory Triggers: Unveiling Memorization in Text-To-Image Generative Models through Word-Level Duplication**

cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03692v1) [paper-pdf](http://arxiv.org/pdf/2312.03692v1)

**Authors**: Ali Naseh, Jaechul Roh, Amir Houmansadr

**Abstract**: Diffusion-based models, such as the Stable Diffusion model, have revolutionized text-to-image synthesis with their ability to produce high-quality, high-resolution images. These advancements have prompted significant progress in image generation and editing tasks. However, these models also raise concerns due to their tendency to memorize and potentially replicate exact training samples, posing privacy risks and enabling adversarial attacks. Duplication in training datasets is recognized as a major factor contributing to memorization, and various forms of memorization have been studied so far. This paper focuses on two distinct and underexplored types of duplication that lead to replication during inference in diffusion-based models, particularly in the Stable Diffusion model. We delve into these lesser-studied duplication phenomena and their implications through two case studies, aiming to contribute to the safer and more responsible use of generative models in various applications.



## **2. PyraTrans: Attention-Enriched Pyramid Transformer for Malicious URL Detection**

cs.CR

12 pages, 7 figures

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.00508v2) [paper-pdf](http://arxiv.org/pdf/2312.00508v2)

**Authors**: Ruitong Liu, Yanbin Wang, Zhenhao Guo, Haitao Xu, Zhan Qin, Wenrui Ma, Fan Zhang

**Abstract**: Although advancements in machine learning have driven the development of malicious URL detection technology, current techniques still face significant challenges in their capacity to generalize and their resilience against evolving threats. In this paper, we propose PyraTrans, a novel method that integrates pretrained Transformers with pyramid feature learning to detect malicious URL. PyraTrans utilizes a pretrained CharBERT as its foundation and is augmented with three interconnected feature modules: 1) Encoder Feature Extraction, extracting multi-order feature matrices from each CharBERT encoder layer; 2) Multi-Scale Feature Learning, capturing local contextual insights at various scales and aggregating information across encoder layers; and 3) Spatial Pyramid Attention, focusing on regional-level attention to emphasize areas rich in expressive information. The proposed approach addresses the limitations of the Transformer in local feature learning and regional relational awareness, which are vital for capturing URL-specific word patterns, character combinations, or structural anomalies. In several challenging experimental scenarios, the proposed method has shown significant improvements in accuracy, generalization, and robustness in malicious URL detection. For instance, it achieved a peak F1-score improvement of 40% in class-imbalanced scenarios, and exceeded the best baseline result by 14.13% in accuracy in adversarial attack scenarios. Additionally, we conduct a case study where our method accurately identifies all 30 active malicious web pages, whereas two pior SOTA methods miss 4 and 7 malicious web pages respectively. Codes and data are available at:https://github.com/Alixyvtte/PyraTrans.



## **3. Defense Against Adversarial Attacks using Convolutional Auto-Encoders**

cs.CV

9 pages, 6 figures, 3 tables

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03520v1) [paper-pdf](http://arxiv.org/pdf/2312.03520v1)

**Authors**: Shreyasi Mandal

**Abstract**: Deep learning models, while achieving state-of-the-art performance on many tasks, are susceptible to adversarial attacks that exploit inherent vulnerabilities in their architectures. Adversarial attacks manipulate the input data with imperceptible perturbations, causing the model to misclassify the data or produce erroneous outputs. This work is based on enhancing the robustness of targeted classifier models against adversarial attacks. To achieve this, an convolutional autoencoder-based approach is employed that effectively counters adversarial perturbations introduced to the input images. By generating images closely resembling the input images, the proposed methodology aims to restore the model's accuracy.



## **4. Quantum-secured single-pixel imaging under general spoofing attack**

quant-ph

9 pages, 6 figures

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03465v1) [paper-pdf](http://arxiv.org/pdf/2312.03465v1)

**Authors**: Jaesung Heo, Taek Jeong, Nam Hun Park, Yonggi Jo

**Abstract**: In this paper, we introduce a quantum-secured single-pixel imaging (QS-SPI) technique designed to withstand spoofing attacks, wherein adversaries attempt to deceive imaging systems with fake signals. Unlike previous quantum-secured protocols that impose a threshold error rate limiting their operation, even with the existence of true signals, our approach not only identifies spoofing attacks but also facilitates the reconstruction of a true image. Our method involves the analysis of a specific mode correlation of a photon-pair, which is independent of the mode used for image construction, to check security. Through this analysis, we can identify both the targeted image region by the attack and the type of spoofing attack, enabling reconstruction of the true image. A proof-of-principle demonstration employing polarization-correlation of a photon-pair is provided, showcasing successful image reconstruction even under the condition of spoofing signals 2000 times stronger than the true signals. We expect our approach to be applied to quantum-secured signal processing such as quantum target detection or ranging.



## **5. Synthesizing Physical Backdoor Datasets: An Automated Framework Leveraging Deep Generative Models**

cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03419v1) [paper-pdf](http://arxiv.org/pdf/2312.03419v1)

**Authors**: Sze Jue Yang, Chinh D. La, Quang H. Nguyen, Eugene Bagdasaryan, Kok-Seng Wong, Anh Tuan Tran, Chee Seng Chan, Khoa D. Doan

**Abstract**: Backdoor attacks, representing an emerging threat to the integrity of deep neural networks, have garnered significant attention due to their ability to compromise deep learning systems clandestinely. While numerous backdoor attacks occur within the digital realm, their practical implementation in real-world prediction systems remains limited and vulnerable to disturbances in the physical world. Consequently, this limitation has given rise to the development of physical backdoor attacks, where trigger objects manifest as physical entities within the real world. However, creating the requisite dataset to train or evaluate a physical backdoor model is a daunting task, limiting the backdoor researchers and practitioners from studying such physical attack scenarios. This paper unleashes a recipe that empowers backdoor researchers to effortlessly create a malicious, physical backdoor dataset based on advances in generative modeling. Particularly, this recipe involves 3 automatic modules: suggesting the suitable physical triggers, generating the poisoned candidate samples (either by synthesizing new samples or editing existing clean samples), and finally refining for the most plausible ones. As such, it effectively mitigates the perceived complexity associated with creating a physical backdoor dataset, transforming it from a daunting task into an attainable objective. Extensive experiment results show that datasets created by our "recipe" enable adversaries to achieve an impressive attack success rate on real physical world data and exhibit similar properties compared to previous physical backdoor attack studies. This paper offers researchers a valuable toolkit for studies of physical backdoors, all within the confines of their laboratories.



## **6. SAIF: Sparse Adversarial and Imperceptible Attack Framework**

cs.CV

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2212.07495v2) [paper-pdf](http://arxiv.org/pdf/2212.07495v2)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.



## **7. Privacy-Preserving Task-Oriented Semantic Communications Against Model Inversion Attacks**

cs.IT

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03252v1) [paper-pdf](http://arxiv.org/pdf/2312.03252v1)

**Authors**: Yanhu Wang, Shuaishuai Guo, Yiqin Deng, Haixia Zhang, Yuguang Fang

**Abstract**: Semantic communication has been identified as a core technology for the sixth generation (6G) of wireless networks. Recently, task-oriented semantic communications have been proposed for low-latency inference with limited bandwidth. Although transmitting only task-related information does protect a certain level of user privacy, adversaries could apply model inversion techniques to reconstruct the raw data or extract useful information, thereby infringing on users' privacy. To mitigate privacy infringement, this paper proposes an information bottleneck and adversarial learning (IBAL) approach to protect users' privacy against model inversion attacks. Specifically, we extract task-relevant features from the input based on the information bottleneck (IB) theory. To overcome the difficulty in calculating the mutual information in high-dimensional space, we derive a variational upper bound to estimate the true mutual information. To prevent data reconstruction from task-related features by adversaries, we leverage adversarial learning to train encoder to fool adversaries by maximizing reconstruction distortion. Furthermore, considering the impact of channel variations on privacy-utility trade-off and the difficulty in manually tuning the weights of each loss, we propose an adaptive weight adjustment method. Numerical results demonstrate that the proposed approaches can effectively protect privacy without significantly affecting task performance and achieve better privacy-utility trade-offs than baseline methods.



## **8. A Simple Framework to Enhance the Adversarial Robustness of Deep Learning-based Intrusion Detection System**

cs.CR

Accepted by Computers & Security

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03245v1) [paper-pdf](http://arxiv.org/pdf/2312.03245v1)

**Authors**: Xinwei Yuan, Shu Han, Wei Huang, Hongliang Ye, Xianglong Kong, Fan Zhang

**Abstract**: Deep learning based intrusion detection systems (DL-based IDS) have emerged as one of the best choices for providing security solutions against various network intrusion attacks. However, due to the emergence and development of adversarial deep learning technologies, it becomes challenging for the adoption of DL models into IDS. In this paper, we propose a novel IDS architecture that can enhance the robustness of IDS against adversarial attacks by combining conventional machine learning (ML) models and Deep Learning models. The proposed DLL-IDS consists of three components: DL-based IDS, adversarial example (AE) detector, and ML-based IDS. We first develop a novel AE detector based on the local intrinsic dimensionality (LID). Then, we exploit the low attack transferability between DL models and ML models to find a robust ML model that can assist us in determining the maliciousness of AEs. If the input traffic is detected as an AE, the ML-based IDS will predict the maliciousness of input traffic, otherwise the DL-based IDS will work for the prediction. The fusion mechanism can leverage the high prediction accuracy of DL models and low attack transferability between DL models and ML models to improve the robustness of the whole system. In our experiments, we observe a significant improvement in the prediction performance of the IDS when subjected to adversarial attack, achieving high accuracy with low resource consumption.



## **9. Model-tuning Via Prompts Makes NLP Models Adversarially Robust**

cs.CL

Accepted to the EMNLP 2023 Conference

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2303.07320v2) [paper-pdf](http://arxiv.org/pdf/2303.07320v2)

**Authors**: Mrigank Raman, Pratyush Maini, J. Zico Kolter, Zachary C. Lipton, Danish Pruthi

**Abstract**: In recent years, NLP practitioners have converged on the following practice: (i) import an off-the-shelf pretrained (masked) language model; (ii) append a multilayer perceptron atop the CLS token's hidden representation (with randomly initialized weights); and (iii) fine-tune the entire model on a downstream task (MLP-FT). This procedure has produced massive gains on standard NLP benchmarks, but these models remain brittle, even to mild adversarial perturbations. In this work, we demonstrate surprising gains in adversarial robustness enjoyed by Model-tuning Via Prompts (MVP), an alternative method of adapting to downstream tasks. Rather than appending an MLP head to make output prediction, MVP appends a prompt template to the input, and makes prediction via text infilling/completion. Across 5 NLP datasets, 4 adversarial attacks, and 3 different models, MVP improves performance against adversarial substitutions by an average of 8% over standard methods and even outperforms adversarial training-based state-of-art defenses by 3.5%. By combining MVP with adversarial training, we achieve further improvements in adversarial robustness while maintaining performance on unperturbed examples. Finally, we conduct ablations to investigate the mechanism underlying these gains. Notably, we find that the main causes of vulnerability of MLP-FT can be attributed to the misalignment between pre-training and fine-tuning tasks, and the randomly initialized MLP parameters.



## **10. Effective Backdoor Mitigation Depends on the Pre-training Objective**

cs.LG

Accepted for oral presentation at BUGS workshop @ NeurIPS 2023  (https://neurips2023-bugs.github.io/)

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2311.14948v3) [paper-pdf](http://arxiv.org/pdf/2311.14948v3)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for pre-training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in these models such as CleanCLIP which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP is ineffective when stronger pre-training objectives are used, even with extensive hyperparameter tuning. Our findings underscore critical considerations for ML practitioners who pre-train models using large-scale web-curated data and are concerned about potential backdoor threats. Notably, our results suggest that simpler pre-training objectives are more amenable to effective backdoor removal. This insight is pivotal for practitioners seeking to balance the trade-offs between using stronger pre-training objectives and security against backdoor attacks.



## **11. Beyond Detection: Unveiling Fairness Vulnerabilities in Abusive Language Models**

cs.CL

Under review

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2311.09428v2) [paper-pdf](http://arxiv.org/pdf/2311.09428v2)

**Authors**: Yueqing Liang, Lu Cheng, Ali Payani, Kai Shu

**Abstract**: This work investigates the potential of undermining both fairness and detection performance in abusive language detection. In a dynamic and complex digital world, it is crucial to investigate the vulnerabilities of these detection models to adversarial fairness attacks to improve their fairness robustness. We propose a simple yet effective framework FABLE that leverages backdoor attacks as they allow targeted control over the fairness and detection performance. FABLE explores three types of trigger designs (i.e., rare, artificial, and natural triggers) and novel sampling strategies. Specifically, the adversary can inject triggers into samples in the minority group with the favored outcome (i.e., "non-abusive") and flip their labels to the unfavored outcome, i.e., "abusive". Experiments on benchmark datasets demonstrate the effectiveness of FABLE attacking fairness and utility in abusive language detection.



## **12. ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**

cs.CV

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03085v1) [paper-pdf](http://arxiv.org/pdf/2312.03085v1)

**Authors**: Xiaohu Lu, Hayder Radha

**Abstract**: The adversarial robustness of a model is its ability to resist adversarial attacks in the form of small perturbations to input data. Universal adversarial attack methods such as Fast Sign Gradient Method (FSGM) and Projected Gradient Descend (PGD) are popular for LiDAR object detection, but they are often deficient compared to task-specific adversarial attacks. Additionally, these universal methods typically require unrestricted access to the model's information, which is difficult to obtain in real-world applications. To address these limitations, we present a black-box Scaling Adversarial Robustness (ScAR) method for LiDAR object detection. By analyzing the statistical characteristics of 3D object detection datasets such as KITTI, Waymo, and nuScenes, we have found that the model's prediction is sensitive to scaling of 3D instances. We propose three black-box scaling adversarial attack methods based on the available information: model-aware attack, distribution-aware attack, and blind attack. We also introduce a strategy for generating scaling adversarial examples to improve the model's robustness against these three scaling adversarial attacks. Comparison with other methods on public datasets under different 3D object detection architectures demonstrates the effectiveness of our proposed method.



## **13. Realistic Scatterer Based Adversarial Attacks on SAR Image Classifiers**

cs.CV

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.02912v1) [paper-pdf](http://arxiv.org/pdf/2312.02912v1)

**Authors**: Tian Ye, Rajgopal Kannan, Viktor Prasanna, Carl Busart, Lance Kaplan

**Abstract**: Adversarial attacks have highlighted the vulnerability of classifiers based on machine learning for Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) tasks. An adversarial attack perturbs SAR images of on-ground targets such that the classifiers are misled into making incorrect predictions. However, many existing attacking techniques rely on arbitrary manipulation of SAR images while overlooking the feasibility of executing the attacks on real-world SAR imagery. Instead, adversarial attacks should be able to be implemented by physical actions, for example, placing additional false objects as scatterers around the on-ground target to perturb the SAR image and fool the SAR ATR.   In this paper, we propose the On-Target Scatterer Attack (OTSA), a scatterer-based physical adversarial attack. To ensure the feasibility of its physical execution, we enforce a constraint on the positioning of the scatterers. Specifically, we restrict the scatterers to be placed only on the target instead of in the shadow regions or the background. To achieve this, we introduce a positioning score based on Gaussian kernels and formulate an optimization problem for our OTSA attack. Using a gradient ascent method to solve the optimization problem, the OTSA can generate a vector of parameters describing the positions, shapes, sizes and amplitudes of the scatterers to guide the physical execution of the attack that will mislead SAR image classifiers. The experimental results show that our attack obtains significantly higher success rates under the positioning constraint compared with the existing method.



## **14. Scaling Laws for Adversarial Attacks on Language Model Activations**

cs.LG

15 pages, 9 figures

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.02780v1) [paper-pdf](http://arxiv.org/pdf/2312.02780v1)

**Authors**: Stanislav Fort

**Abstract**: We explore a class of adversarial attacks targeting the activations of language models. By manipulating a relatively small subset of model activations, $a$, we demonstrate the ability to control the exact prediction of a significant number (in some cases up to 1000) of subsequent tokens $t$. We empirically verify a scaling law where the maximum number of target tokens $t_\mathrm{max}$ predicted depends linearly on the number of tokens $a$ whose activations the attacker controls as $t_\mathrm{max} = \kappa a$. We find that the number of bits of control in the input space needed to control a single bit in the output space (what we call attack resistance $\chi$) is remarkably constant between $\approx 16$ and $\approx 25$ over 2 orders of magnitude of model sizes for different language models. Compared to attacks on tokens, attacks on activations are predictably much stronger, however, we identify a surprising regularity where one bit of input steered either via activations or via tokens is able to exert control over a similar amount of output bits. This gives support for the hypothesis that adversarial attacks are a consequence of dimensionality mismatch between the input and output spaces. A practical implication of the ease of attacking language model activations instead of tokens is for multi-modal and selected retrieval models, where additional data sources are added as activations directly, sidestepping the tokenized input. This opens up a new, broad attack surface. By using language models as a controllable test-bed to study adversarial attacks, we were able to experiment with input-output dimensions that are inaccessible in computer vision, especially where the output dimension dominates.



## **15. Generating Visually Realistic Adversarial Patch**

cs.CV

14 pages

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03030v1) [paper-pdf](http://arxiv.org/pdf/2312.03030v1)

**Authors**: Xiaosen Wang, Kunyu Wang

**Abstract**: Deep neural networks (DNNs) are vulnerable to various types of adversarial examples, bringing huge threats to security-critical applications. Among these, adversarial patches have drawn increasing attention due to their good applicability to fool DNNs in the physical world. However, existing works often generate patches with meaningless noise or patterns, making it conspicuous to humans. To address this issue, we explore how to generate visually realistic adversarial patches to fool DNNs. Firstly, we analyze that a high-quality adversarial patch should be realistic, position irrelevant, and printable to be deployed in the physical world. Based on this analysis, we propose an effective attack called VRAP, to generate visually realistic adversarial patches. Specifically, VRAP constrains the patch in the neighborhood of a real image to ensure the visual reality, optimizes the patch at the poorest position for position irrelevance, and adopts Total Variance loss as well as gamma transformation to make the generated patch printable without losing information. Empirical evaluations on the ImageNet dataset demonstrate that the proposed VRAP exhibits outstanding attack performance in the digital world. Moreover, the generated adversarial patches can be disguised as the scrawl or logo in the physical world to fool the deep models without being detected, bringing significant threats to DNNs-enabled applications.



## **16. Byzantine-Robust Distributed Online Learning: Taming Adversarial Participants in An Adversarial Environment**

cs.LG

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2307.07980v3) [paper-pdf](http://arxiv.org/pdf/2307.07980v3)

**Authors**: Xingrong Dong, Zhaoxian Wu, Qing Ling, Zhi Tian

**Abstract**: This paper studies distributed online learning under Byzantine attacks. The performance of an online learning algorithm is often characterized by (adversarial) regret, which evaluates the quality of one-step-ahead decision-making when an environment provides adversarial losses, and a sublinear bound is preferred. But we prove that, even with a class of state-of-the-art robust aggregation rules, in an adversarial environment and in the presence of Byzantine participants, distributed online gradient descent can only achieve a linear adversarial regret bound, which is tight. This is the inevitable consequence of Byzantine attacks, even though we can control the constant of the linear adversarial regret to a reasonable level. Interestingly, when the environment is not fully adversarial so that the losses of the honest participants are i.i.d. (independent and identically distributed), we show that sublinear stochastic regret, in contrast to the aforementioned adversarial regret, is possible. We develop a Byzantine-robust distributed online momentum algorithm to attain such a sublinear stochastic regret bound. Extensive numerical experiments corroborate our theoretical analysis.



## **17. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01886v1) [paper-pdf](http://arxiv.org/pdf/2312.01886v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.



## **18. Two-stage optimized unified adversarial patch for attacking visible-infrared cross-modal detectors in the physical world**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01789v1) [paper-pdf](http://arxiv.org/pdf/2312.01789v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Currently, many studies have addressed security concerns related to visible and infrared detectors independently. In practical scenarios, utilizing cross-modal detectors for tasks proves more reliable than relying on single-modal detectors. Despite this, there is a lack of comprehensive security evaluations for cross-modal detectors. While existing research has explored the feasibility of attacks against cross-modal detectors, the implementation of a robust attack remains unaddressed. This work introduces the Two-stage Optimized Unified Adversarial Patch (TOUAP) designed for performing attacks against visible-infrared cross-modal detectors in real-world, black-box settings. The TOUAP employs a two-stage optimization process: firstly, PSO optimizes an irregular polygonal infrared patch to attack the infrared detector; secondly, the color QR code is optimized, and the shape information of the infrared patch from the first stage is used as a mask. The resulting irregular polygon visible modal patch executes an attack on the visible detector. Through extensive experiments conducted in both digital and physical environments, we validate the effectiveness and robustness of the proposed method. As the TOUAP surpasses baseline performance, we advocate for its widespread attention.



## **19. Singular Regularization with Information Bottleneck Improves Model's Adversarial Robustness**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02237v1) [paper-pdf](http://arxiv.org/pdf/2312.02237v1)

**Authors**: Guanlin Li, Naishan Zheng, Man Zhou, Jie Zhang, Tianwei Zhang

**Abstract**: Adversarial examples are one of the most severe threats to deep learning models. Numerous works have been proposed to study and defend adversarial examples. However, these works lack analysis of adversarial information or perturbation, which cannot reveal the mystery of adversarial examples and lose proper interpretation. In this paper, we aim to fill this gap by studying adversarial information as unstructured noise, which does not have a clear pattern. Specifically, we provide some empirical studies with singular value decomposition, by decomposing images into several matrices, to analyze adversarial information for different attacks. Based on the analysis, we propose a new module to regularize adversarial information and combine information bottleneck theory, which is proposed to theoretically restrict intermediate representations. Therefore, our method is interpretable. Moreover, the fashion of our design is a novel principle that is general and unified. Equipped with our new module, we evaluate two popular model structures on two mainstream datasets with various adversarial attacks. The results indicate that the improvement in robust accuracy is significant. On the other hand, we prove that our method is efficient with only a few additional parameters and able to be explained under regional faithfulness analysis.



## **20. Warfare:Breaking the Watermark Protection of AI-Generated Content**

cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2310.07726v2) [paper-pdf](http://arxiv.org/pdf/2310.07726v2)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.



## **21. Malicious Lateral Movement in 5G Core With Network Slicing And Its Detection**

cs.CR

Accepted for publication in the Proceedings of IEEE ITNAC-2023

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01681v1) [paper-pdf](http://arxiv.org/pdf/2312.01681v1)

**Authors**: Ayush Kumar, Vrizlynn L. L. Thing

**Abstract**: 5G networks are susceptible to cyber attacks due to reasons such as implementation issues and vulnerabilities in 3GPP standard specifications. In this work, we propose lateral movement strategies in a 5G Core (5GC) with network slicing enabled, as part of a larger attack campaign by well-resourced adversaries such as APT groups. Further, we present 5GLatte, a system to detect such malicious lateral movement. 5GLatte operates on a host-container access graph built using host/NF container logs collected from the 5GC. Paths inferred from the access graph are scored based on selected filtering criteria and subsequently presented as input to a threshold-based anomaly detection algorithm to reveal malicious lateral movement paths. We evaluate 5GLatte on a dataset containing attack campaigns (based on MITRE ATT&CK and FiGHT frameworks) launched in a 5G test environment which shows that compared to other lateral movement detectors based on state-of-the-art, it can achieve higher true positive rates with similar false positive rates.



## **22. Adversarial Medical Image with Hierarchical Feature Hiding**

eess.IV

Our code is available at  \url{https://github.com/qsyao/Hierarchical_Feature_Constraint}. arXiv admin  note: text overlap with arXiv:2012.09501

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01679v1) [paper-pdf](http://arxiv.org/pdf/2312.01679v1)

**Authors**: Qingsong Yao, Zecheng He, Yuexiang Li, Yi Lin, Kai Ma, Yefeng Zheng, S. Kevin Zhou

**Abstract**: Deep learning based methods for medical images can be easily compromised by adversarial examples (AEs), posing a great security flaw in clinical decision-making. It has been discovered that conventional adversarial attacks like PGD which optimize the classification logits, are easy to distinguish in the feature space, resulting in accurate reactive defenses. To better understand this phenomenon and reassess the reliability of the reactive defenses for medical AEs, we thoroughly investigate the characteristic of conventional medical AEs. Specifically, we first theoretically prove that conventional adversarial attacks change the outputs by continuously optimizing vulnerable features in a fixed direction, thereby leading to outlier representations in the feature space. Then, a stress test is conducted to reveal the vulnerability of medical images, by comparing with natural images. Interestingly, this vulnerability is a double-edged sword, which can be exploited to hide AEs. We then propose a simple-yet-effective hierarchical feature constraint (HFC), a novel add-on to conventional white-box attacks, which assists to hide the adversarial feature in the target feature distribution. The proposed method is evaluated on three medical datasets, both 2D and 3D, with different modalities. The experimental results demonstrate the superiority of HFC, \emph{i.e.,} it bypasses an array of state-of-the-art adversarial medical AE detectors more efficiently than competing adaptive attacks, which reveals the deficiencies of medical reactive defense and allows to develop more robust defenses in future.



## **23. The Queen's Guard: A Secure Enforcement of Fine-grained Access Control In Distributed Data Analytics Platforms**

cs.CR

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2106.13123v4) [paper-pdf](http://arxiv.org/pdf/2106.13123v4)

**Authors**: Fahad Shaon, Sazzadur Rahaman, Murat Kantarcioglu

**Abstract**: Distributed data analytics platforms (i.e., Apache Spark, Hadoop) provide high-level APIs to programmatically write analytics tasks that are run distributedly in multiple computing nodes. The design of these frameworks was primarily motivated by performance and usability. Thus, the security takes a back seat. Consequently, they do not inherently support fine-grained access control or offer any plugin mechanism to enable it, making them risky to be used in multi-tier organizational settings.   There have been attempts to build "add-on" solutions to enable fine-grained access control for distributed data analytics platforms. In this paper, first, we show that straightforward enforcement of ``add-on'' access control is insecure under adversarial code execution. Specifically, we show that an attacker can abuse platform-provided APIs to evade access controls without leaving any traces. Second, we designed a two-layered (i.e., proactive and reactive) defense system to protect against API abuses. On submission of a user code, our proactive security layer statically screens it to find potential attack signatures prior to its execution. The reactive security layer employs code instrumentation-based runtime checks and sandboxed execution to throttle any exploits at runtime. Next, we propose a new fine-grained access control framework with an enhanced policy language that supports map and filter primitives. Finally, we build a system named SecureDL with our new access control framework and defense system on top of Apache Spark, which ensures secure access control policy enforcement under adversaries capable of executing code.   To the best of our knowledge, this is the first fine-grained attribute-based access control framework for distributed data analytics platforms that is secure against platform API abuse attacks. Performance evaluation showed that the overhead due to added security is low.



## **24. Robust Evaluation of Diffusion-Based Adversarial Purification**

cs.CV

Accepted by ICCV 2023, oral presentation. Code is available at  https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2303.09051v3) [paper-pdf](http://arxiv.org/pdf/2303.09051v3)

**Authors**: Minjong Lee, Dongwoo Kim

**Abstract**: We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy improving robustness compared to the current diffusion-based purification methods.



## **25. QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers**

cs.CV

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.02220v1) [paper-pdf](http://arxiv.org/pdf/2312.02220v1)

**Authors**: Amit Baras, Alon Zolfi, Yuval Elovici, Asaf Shabtai

**Abstract**: In recent years, there has been a significant trend in deep neural networks (DNNs), particularly transformer-based models, of developing ever-larger and more capable models. While they demonstrate state-of-the-art performance, their growing scale requires increased computational resources (e.g., GPUs with greater memory capacity). To address this problem, quantization techniques (i.e., low-bit-precision representation and matrix multiplication) have been proposed. Most quantization techniques employ a static strategy in which the model parameters are quantized, either during training or inference, without considering the test-time sample. In contrast, dynamic quantization techniques, which have become increasingly popular, adapt during inference based on the input provided, while maintaining full-precision performance. However, their dynamic behavior and average-case performance assumption makes them vulnerable to a novel threat vector -- adversarial attacks that target the model's efficiency and availability. In this paper, we present QuantAttack, a novel attack that targets the availability of quantized models, slowing down the inference, and increasing memory usage and energy consumption. We show that carefully crafted adversarial examples, which are designed to exhaust the resources of the operating system, can trigger worst-case performance. In our experiments, we demonstrate the effectiveness of our attack on vision transformers on a wide range of tasks, both uni-modal and multi-modal. We also examine the effect of different attack variants (e.g., a universal perturbation) and the transferability between different models.



## **26. Exploring Adversarial Robustness of LiDAR-Camera Fusion Model in Autonomous Driving**

cs.RO

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01468v1) [paper-pdf](http://arxiv.org/pdf/2312.01468v1)

**Authors**: Bo Yang, Xiaoyu Ji, Xiaoyu Ji, Xiaoyu Ji, Xiaoyu Ji

**Abstract**: Our study assesses the adversarial robustness of LiDAR-camera fusion models in 3D object detection. We introduce an attack technique that, by simply adding a limited number of physically constrained adversarial points above a car, can make the car undetectable by the fusion model. Experimental results reveal that even without changes to the image data channel, the fusion model can be deceived solely by manipulating the LiDAR data channel. This finding raises safety concerns in the field of autonomous driving. Further, we explore how the quantity of adversarial points, the distance between the front-near car and the LiDAR-equipped car, and various angular factors affect the attack success rate. We believe our research can contribute to the understanding of multi-sensor robustness, offering insights and guidance to enhance the safety of autonomous driving.



## **27. Evaluating the Security of Satellite Systems**

cs.CR

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01330v1) [paper-pdf](http://arxiv.org/pdf/2312.01330v1)

**Authors**: Roy Peled, Eran Aizikovich, Edan Habler, Yuval Elovici, Asaf Shabtai

**Abstract**: Satellite systems are facing an ever-increasing amount of cybersecurity threats as their role in communications, navigation, and other services expands. Recent papers have examined attacks targeting satellites and space systems; however, they did not comprehensively analyze the threats to satellites and systematically identify adversarial techniques across the attack lifecycle. This paper presents a comprehensive taxonomy of adversarial tactics, techniques, and procedures explicitly targeting LEO satellites. First, we analyze the space ecosystem including the ground, space, Communication, and user segments, highlighting their architectures, functions, and vulnerabilities. Then, we examine the threat landscape, including adversary types, and capabilities, and survey historical and recent attacks such as jamming, spoofing, and supply chain. Finally, we propose a novel extension of the MITRE ATT&CK framework to categorize satellite attack techniques across the adversary lifecycle from reconnaissance to impact. The taxonomy is demonstrated by modeling high-profile incidents, including the Viasat attack that disrupted Ukraine's communications. The taxonomy provides the foundation for the development of defenses against emerging cyber risks to space assets. The proposed threat model will advance research in the space domain and contribute to the security of the space domain against sophisticated attacks.



## **28. Rethinking PGD Attack: Is Sign Function Necessary?**

cs.LG

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01260v1) [paper-pdf](http://arxiv.org/pdf/2312.01260v1)

**Authors**: Junjie Yang, Tianlong Chen, Xuxi Chen, Zhangyang Wang, Yingbin Liang

**Abstract**: Neural networks have demonstrated success in various domains, yet their performance can be significantly degraded by even a small input perturbation. Consequently, the construction of such perturbations, known as adversarial attacks, has gained significant attention, many of which fall within "white-box" scenarios where we have full access to the neural network. Existing attack algorithms, such as the projected gradient descent (PGD), commonly take the sign function on the raw gradient before updating adversarial inputs, thereby neglecting gradient magnitude information. In this paper, we present a theoretical analysis of how such sign-based update algorithm influences step-wise attack performance, as well as its caveat. We also interpret why previous attempts of directly using raw gradients failed. Based on that, we further propose a new raw gradient descent (RGD) algorithm that eliminates the use of sign. Specifically, we convert the constrained optimization problem into an unconstrained one, by introducing a new hidden variable of non-clipped perturbation that can move beyond the constraint. The effectiveness of the proposed RGD algorithm has been demonstrated extensively in experiments, outperforming PGD and other competitors in various settings, without incurring any additional computational overhead. The codes is available in https://github.com/JunjieYang97/RGD.



## **29. TranSegPGD: Improving Transferability of Adversarial Examples on Semantic Segmentation**

cs.CV

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.02207v1) [paper-pdf](http://arxiv.org/pdf/2312.02207v1)

**Authors**: Xiaojun Jia, Jindong Gu, Yihao Huang, Simeng Qin, Qing Guo, Yang Liu, Xiaochun Cao

**Abstract**: Transferability of adversarial examples on image classification has been systematically explored, which generates adversarial examples in black-box mode. However, the transferability of adversarial examples on semantic segmentation has been largely overlooked. In this paper, we propose an effective two-stage adversarial attack strategy to improve the transferability of adversarial examples on semantic segmentation, dubbed TranSegPGD. Specifically, at the first stage, every pixel in an input image is divided into different branches based on its adversarial property. Different branches are assigned different weights for optimization to improve the adversarial performance of all pixels.We assign high weights to the loss of the hard-to-attack pixels to misclassify all pixels. At the second stage, the pixels are divided into different branches based on their transferable property which is dependent on Kullback-Leibler divergence. Different branches are assigned different weights for optimization to improve the transferability of the adversarial examples. We assign high weights to the loss of the high-transferability pixels to improve the transferability of adversarial examples. Extensive experiments with various segmentation models are conducted on PASCAL VOC 2012 and Cityscapes datasets to demonstrate the effectiveness of the proposed method. The proposed adversarial attack method can achieve state-of-the-art performance.



## **30. Look Closer to Your Enemy: Learning to Attack via Teacher-Student Mimicking**

cs.CV

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2207.13381v4) [paper-pdf](http://arxiv.org/pdf/2207.13381v4)

**Authors**: Mingjie Wang, Jianxiong Guo, Sirui Li, Dingwen Xiao, Zhiqing Tang

**Abstract**: Deep neural networks have significantly advanced person re-identification (ReID) applications in the realm of the industrial internet, yet they remain vulnerable. Thus, it is crucial to study the robustness of ReID systems, as there are risks of adversaries using these vulnerabilities to compromise industrial surveillance systems. Current adversarial methods focus on generating attack samples using misclassification feedback from victim models (VMs), neglecting VM's cognitive processes. We seek to address this by producing authentic ReID attack instances through VM cognition decryption. This approach boasts advantages like better transferability to open-set ReID tests, easier VM misdirection, and enhanced creation of realistic and undetectable assault images. However, the task of deciphering the cognitive mechanism in VM is widely considered to be a formidable challenge. In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE (Look Closer to Your Enemy), to generate adversarial query images. Specifically, LCYE first distills VM's knowledge via teacher-student memory mimicking the proxy task. This knowledge prior serves as an unambiguous cryptographic token, encapsulating elements deemed indispensable and plausible by the VM, with the intent of facilitating precise adversarial misdirection. Further, benefiting from the multiple opposing task framework of LCYE, we investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. The source code can be found at https://github.com/MingjieWang0606/LCYE-attack_reid.



## **31. FRAUDability: Estimating Users' Susceptibility to Financial Fraud Using Adversarial Machine Learning**

cs.CR

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.01200v1) [paper-pdf](http://arxiv.org/pdf/2312.01200v1)

**Authors**: Chen Doytshman, Satoru Momiyama, Inderjeet Singh, Yuval Elovici, Asaf Shabtai

**Abstract**: In recent years, financial fraud detection systems have become very efficient at detecting fraud, which is a major threat faced by e-commerce platforms. Such systems often include machine learning-based algorithms aimed at detecting and reporting fraudulent activity. In this paper, we examine the application of adversarial learning based ranking techniques in the fraud detection domain and propose FRAUDability, a method for the estimation of a financial fraud detection system's performance for every user. We are motivated by the assumption that "not all users are created equal" -- while some users are well protected by fraud detection algorithms, others tend to pose a challenge to such systems. The proposed method produces scores, namely "fraudability scores," which are numerical estimations of a fraud detection system's ability to detect financial fraud for a specific user, given his/her unique activity in the financial system. Our fraudability scores enable those tasked with defending users in a financial platform to focus their attention and resources on users with high fraudability scores to better protect them. We validate our method using a real e-commerce platform's dataset and demonstrate the application of fraudability scores from the attacker's perspective, on the platform, and more specifically, on the fraud detection systems used by the e-commerce enterprise. We show that the scores can also help attackers increase their financial profit by 54%, by engaging solely with users with high fraudability scores, avoiding those users whose spending habits enable more accurate fraud detection.



## **32. Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions**

cs.LG

Published as a conference paper at NeurIPS 2023

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2306.15427v2) [paper-pdf](http://arxiv.org/pdf/2306.15427v2)

**Authors**: Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

**Abstract**: Despite its success in the image domain, adversarial training did not (yet) stand out as an effective defense for Graph Neural Networks (GNNs) against graph structure perturbations. In the pursuit of fixing adversarial training (1) we show and overcome fundamental theoretical as well as practical limitations of the adopted graph learning setting in prior work; (2) we reveal that more flexible GNNs based on learnable graph diffusion are able to adjust to adversarial perturbations, while the learned message passing scheme is naturally interpretable; (3) we introduce the first attack for structure perturbations that, while targeting multiple nodes at once, is capable of handling global (graph-level) as well as local (node-level) constraints. Including these contributions, we demonstrate that adversarial training is a state-of-the-art defense against adversarial structure perturbations.



## **33. Scrappy: SeCure Rate Assuring Protocol with PrivacY**

cs.CR

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.00989v1) [paper-pdf](http://arxiv.org/pdf/2312.00989v1)

**Authors**: Kosei Akama, Yoshimichi Nakatsuka, Masaaki Sato, Keisuke Uehara

**Abstract**: Preventing abusive activities caused by adversaries accessing online services at a rate exceeding that expected by websites has become an ever-increasing problem. CAPTCHAs and SMS authentication are widely used to provide a solution by implementing rate limiting, although they are becoming less effective, and some are considered privacy-invasive. In light of this, many studies have proposed better rate-limiting systems that protect the privacy of legitimate users while blocking malicious actors. However, they suffer from one or more shortcomings: (1) assume trust in the underlying hardware and (2) are vulnerable to side-channel attacks. Motivated by the aforementioned issues, this paper proposes Scrappy: SeCure Rate Assuring Protocol with PrivacY. Scrappy allows clients to generate unforgeable yet unlinkable rate-assuring proofs, which provides the server with cryptographic guarantees that the client is not misbehaving. We design Scrappy using a combination of DAA and hardware security devices. Scrappy is implemented over three types of devices, including one that can immediately be deployed in the real world. Our baseline evaluation shows that the end-to-end latency of Scrappy is minimal, taking only 0.32 seconds, and uses only 679 bytes of bandwidth when transferring necessary data. We also conduct an extensive security evaluation, showing that the rate-limiting capability of Scrappy is unaffected even if the hardware security device is compromised.



## **34. Deep Generative Attacks and Countermeasures for Data-Driven Offline Signature Verification**

cs.CV

10 pages, 7 figures, 1 table, Signature verification, Deep generative  models, attacks, generative attack explainability, data-driven verification  system

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.00987v1) [paper-pdf](http://arxiv.org/pdf/2312.00987v1)

**Authors**: An Ngo, MinhPhuong Cao, Rajesh Kumar

**Abstract**: While previous studies have explored attacks via random, simple, and skilled forgeries, generative attacks have received limited attention in the data-driven signature verification (DASV) process. Thus, this paper explores the impact of generative attacks on DASV and proposes practical and interpretable countermeasures. We investigate the power of two prominent Deep Generative Models (DGMs), Variational Auto-encoders (VAE) and Conditional Generative Adversarial Networks (CGAN), on their ability to generate signatures that would successfully deceive DASV. Additionally, we evaluate the quality of generated images using the Structural Similarity Index measure (SSIM) and use the same to explain the attack's success. Finally, we propose countermeasures that effectively reduce the impact of deep generative attacks on DASV.   We first generated six synthetic datasets from three benchmark offline-signature datasets viz. CEDAR, BHSig260- Bengali, and BHSig260-Hindi using VAE and CGAN. Then, we built baseline DASVs using Xception, ResNet152V2, and DenseNet201. These DASVs achieved average (over the three datasets) False Accept Rates (FARs) of 2.55%, 3.17%, and 1.06%, respectively. Then, we attacked these baselines using the synthetic datasets. The VAE-generated signatures increased average FARs to 10.4%, 10.1%, and 7.5%, while CGAN-generated signatures to 32.5%, 30%, and 26.1%. The variation in the effectiveness of attack for VAE and CGAN was investigated further and explained by a strong (rho = -0.86) negative correlation between FARs and SSIMs. We created another set of synthetic datasets and used the same to retrain the DASVs. The retained baseline showed significant robustness to random, skilled, and generative attacks as the FARs shrank to less than 1% on average. The findings underscore the importance of studying generative attacks and potential countermeasures for DASV.



## **35. Stealing the Decoding Algorithms of Language Models**

cs.LG

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2303.04729v4) [paper-pdf](http://arxiv.org/pdf/2303.04729v4)

**Authors**: Ali Naseh, Kalpesh Krishna, Mohit Iyyer, Amir Houmansadr

**Abstract**: A key component of generating text from modern language models (LM) is the selection and tuning of decoding algorithms. These algorithms determine how to generate text from the internal probability distribution generated by the LM. The process of choosing a decoding algorithm and tuning its hyperparameters takes significant time, manual effort, and computation, and it also requires extensive human evaluation. Therefore, the identity and hyperparameters of such decoding algorithms are considered to be extremely valuable to their owners. In this work, we show, for the first time, that an adversary with typical API access to an LM can steal the type and hyperparameters of its decoding algorithms at very low monetary costs. Our attack is effective against popular LMs used in text generation APIs, including GPT-2, GPT-3 and GPT-Neo. We demonstrate the feasibility of stealing such information with only a few dollars, e.g., $\$0.8$, $\$1$, $\$4$, and $\$40$ for the four versions of GPT-3.



## **36. Adversarial Attacks and Defenses on 3D Point Cloud Classification: A Survey**

cs.CV

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2307.00309v2) [paper-pdf](http://arxiv.org/pdf/2307.00309v2)

**Authors**: Hanieh Naderi, Ivan V. Bajić

**Abstract**: Deep learning has successfully solved a wide range of tasks in 2D vision as a dominant AI technique. Recently, deep learning on 3D point clouds is becoming increasingly popular for addressing various tasks in this field. Despite remarkable achievements, deep learning algorithms are vulnerable to adversarial attacks. These attacks are imperceptible to the human eye but can easily fool deep neural networks in the testing and deployment stage. To encourage future research, this survey summarizes the current progress on adversarial attack and defense techniques on point cloud classification.This paper first introduces the principles and characteristics of adversarial attacks and summarizes and analyzes adversarial example generation methods in recent years. Additionally, it provides an overview of defense strategies, organized into data-focused and model-focused methods. Finally, it presents several current challenges and potential future research directions in this domain.



## **37. A Unified Approach to Interpreting and Boosting Adversarial Transferability**

cs.LG

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2010.04055v2) [paper-pdf](http://arxiv.org/pdf/2010.04055v2)

**Authors**: Xin Wang, Jie Ren, Shuyun Lin, Xiangming Zhu, Yisen Wang, Quanshi Zhang

**Abstract**: In this paper, we use the interaction inside adversarial perturbations to explain and boost the adversarial transferability. We discover and prove the negative correlation between the adversarial transferability and the interaction inside adversarial perturbations. The negative correlation is further verified through different DNNs with various inputs. Moreover, this negative correlation can be regarded as a unified perspective to understand current transferability-boosting methods. To this end, we prove that some classic methods of enhancing the transferability essentially decease interactions inside adversarial perturbations. Based on this, we propose to directly penalize interactions during the attacking process, which significantly improves the adversarial transferability.



## **38. Unleashing Cheapfakes through Trojan Plugins of Large Language Models**

cs.CR

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2312.00374v1) [paper-pdf](http://arxiv.org/pdf/2312.00374v1)

**Authors**: Tian Dong, Guoxing Chen, Shaofeng Li, Minhui Xue, Rayne Holland, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. Our experiments validate that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserves or improves the adapter's utility. Finally, we provide two case studies to demonstrate that the Trojan adapter can lead a LLM-powered autonomous agent to execute unintended scripts or send phishing emails. Our novel attacks represent the first study of supply chain threats for LLMs through the lens of Trojan plugins.



## **39. Bayesian Learning with Information Gain Provably Bounds Risk for a Robust Adversarial Defense**

cs.LG

Published at ICML 2022. Code is available at  https://github.com/baogiadoan/IG-BNN

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2212.02003v2) [paper-pdf](http://arxiv.org/pdf/2212.02003v2)

**Authors**: Bao Gia Doan, Ehsan Abbasnejad, Javen Qinfeng Shi, Damith C. Ranasinghe

**Abstract**: We present a new algorithm to learn a deep neural network model robust against adversarial attacks. Previous algorithms demonstrate an adversarially trained Bayesian Neural Network (BNN) provides improved robustness. We recognize the adversarial learning approach for approximating the multi-modal posterior distribution of a Bayesian model can lead to mode collapse; consequently, the model's achievements in robustness and performance are sub-optimal. Instead, we first propose preventing mode collapse to better approximate the multi-modal posterior distribution. Second, based on the intuition that a robust model should ignore perturbations and only consider the informative content of the input, we conceptualize and formulate an information gain objective to measure and force the information learned from both benign and adversarial training instances to be similar. Importantly. we prove and demonstrate that minimizing the information gain objective allows the adversarial risk to approach the conventional empirical risk. We believe our efforts provide a step toward a basis for a principled method of adversarially training BNNs. Our model demonstrate significantly improved robustness--up to 20%--compared with adversarial training and Adv-BNN under PGD attacks with 0.035 distortion on both CIFAR-10 and STL-10 datasets.



## **40. Security Defense of Large Scale Networks Under False Data Injection Attacks: An Attack Detection Scheduling Approach**

eess.SY

14 pages, 11 figures

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2212.05500v3) [paper-pdf](http://arxiv.org/pdf/2212.05500v3)

**Authors**: Yuhan Suo, Senchun Chai, Runqi Chai, Zhong-Hua Pang, Yuanqing Xia, Guo-Ping Liu

**Abstract**: In large-scale networks, communication links between nodes are easily injected with false data by adversaries. This paper proposes a novel security defense strategy from the perspective of attack detection scheduling to ensure the security of the network. Based on the proposed strategy, each sensor can directly exclude suspicious sensors from its neighboring set. First, the problem of selecting suspicious sensors is formulated as a combinatorial optimization problem, which is non-deterministic polynomial-time hard (NP-hard). To solve this problem, the original function is transformed into a submodular function. Then, we propose a distributed attack detection scheduling algorithm based on the sequential submodular optimization theory, which incorporates \emph{expert problem} to better utilize historical information to guide the sensor selection task at the current moment. For different attack strategies, theoretical results show that the average optimization rate of the proposed algorithm has a lower bound, and the error expectation for any subset is bounded. In addition, under two kinds of insecurity conditions, the proposed algorithm can guarantee the security of the entire network from the perspective of the augmented estimation error. Finally, the effectiveness of the proposed method is verified by the numerical simulation and practical experiment.



## **41. SPAM: Secure & Private Aircraft Management**

cs.CR

6 pages

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00245v1) [paper-pdf](http://arxiv.org/pdf/2312.00245v1)

**Authors**: Yaman Jandali, Nojan Sheybani, Farinaz Koushanfar

**Abstract**: With the rising use of aircrafts for operations ranging from disaster-relief to warfare, there is a growing risk of adversarial attacks. Malicious entities often only require the location of the aircraft for these attacks. Current satellite-aircraft communication and tracking protocols put aircrafts at risk if the satellite is compromised, due to computation being done in plaintext. In this work, we present \texttt{SPAM}, a private, secure, and accurate system that allows satellites to efficiently manage and maintain tracking angles for aircraft fleets without learning aircrafts' locations. \texttt{SPAM} is built upon multi-party computation and zero-knowledge proofs to guarantee privacy and high efficiency. While catered towards aircrafts, \texttt{SPAM}'s zero-knowledge fleet management can be easily extended to the IoT, with very little overhead.



## **42. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16119v2) [paper-pdf](http://arxiv.org/pdf/2311.16119v2)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.



## **43. Optimal Attack and Defense for Reinforcement Learning**

cs.LG

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00198v1) [paper-pdf](http://arxiv.org/pdf/2312.00198v1)

**Authors**: Jeremy McMahan, Young Wu, Xiaojin Zhu, Qiaomin Xie

**Abstract**: To ensure the usefulness of Reinforcement Learning (RL) in real systems, it is crucial to ensure they are robust to noise and adversarial attacks. In adversarial RL, an external attacker has the power to manipulate the victim agent's interaction with the environment. We study the full class of online manipulation attacks, which include (i) state attacks, (ii) observation attacks (which are a generalization of perceived-state attacks), (iii) action attacks, and (iv) reward attacks. We show the attacker's problem of designing a stealthy attack that maximizes its own expected reward, which often corresponds to minimizing the victim's value, is captured by a Markov Decision Process (MDP) that we call a meta-MDP since it is not the true environment but a higher level environment induced by the attacked interaction. We show that the attacker can derive optimal attacks by planning in polynomial time or learning with polynomial sample complexity using standard RL techniques. We argue that the optimal defense policy for the victim can be computed as the solution to a stochastic Stackelberg game, which can be further simplified into a partially-observable turn-based stochastic game (POTBSG). Neither the attacker nor the victim would benefit from deviating from their respective optimal policies, thus such solutions are truly robust. Although the defense problem is NP-hard, we show that optimal Markovian defenses can be computed (learned) in polynomial time (sample complexity) in many scenarios.



## **44. Fool the Hydra: Adversarial Attacks against Multi-view Object Detection Systems**

cs.CV

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00173v1) [paper-pdf](http://arxiv.org/pdf/2312.00173v1)

**Authors**: Bilel Tarchoun, Quazi Mishkatul Alam, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstract**: Adversarial patches exemplify the tangible manifestation of the threat posed by adversarial attacks on Machine Learning (ML) models in real-world scenarios. Robustness against these attacks is of the utmost importance when designing computer vision applications, especially for safety-critical domains such as CCTV systems. In most practical situations, monitoring open spaces requires multi-view systems to overcome acquisition challenges such as occlusion handling. Multiview object systems are able to combine data from multiple views, and reach reliable detection results even in difficult environments. Despite its importance in real-world vision applications, the vulnerability of multiview systems to adversarial patches is not sufficiently investigated. In this paper, we raise the following question: Does the increased performance and information sharing across views offer as a by-product robustness to adversarial patches? We first conduct a preliminary analysis showing promising robustness against off-the-shelf adversarial patches, even in an extreme setting where we consider patches applied to all views by all persons in Wildtrack benchmark. However, we challenged this observation by proposing two new attacks: (i) In the first attack, targeting a multiview CNN, we maximize the global loss by proposing gradient projection to the different views and aggregating the obtained local gradients. (ii) In the second attack, we focus on a Transformer-based multiview framework. In addition to the focal loss, we also maximize the transformer-specific loss by dissipating its attention blocks. Our results show a large degradation in the detection performance of victim multiview systems with our first patch attack reaching an attack success rate of 73% , while our second proposed attack reduced the performance of its target detector by 62%



## **45. Universal Backdoor Attacks**

cs.LG

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00157v1) [paper-pdf](http://arxiv.org/pdf/2312.00157v1)

**Authors**: Benjamin Schneider, Nils Lukas, Florian Kerschbaum

**Abstract**: Web-scraped datasets are vulnerable to data poisoning, which can be used for backdooring deep image classifiers during training. Since training on large datasets is expensive, a model is trained once and re-used many times. Unlike adversarial examples, backdoor attacks often target specific classes rather than any class learned by the model. One might expect that targeting many classes through a naive composition of attacks vastly increases the number of poison samples. We show this is not necessarily true and more efficient, universal data poisoning attacks exist that allow controlling misclassifications from any source class into any target class with a small increase in poison samples. Our idea is to generate triggers with salient characteristics that the model can learn. The triggers we craft exploit a phenomenon we call inter-class poison transferability, where learning a trigger from one class makes the model more vulnerable to learning triggers for other classes. We demonstrate the effectiveness and robustness of our universal backdoor attacks by controlling models with up to 6,000 classes while poisoning only 0.15% of the training dataset.



## **46. On the Adversarial Robustness of Graph Contrastive Learning Methods**

cs.LG

Accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop  (NeurIPS GLFrontiers 2023)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.17853v2) [paper-pdf](http://arxiv.org/pdf/2311.17853v2)

**Authors**: Filippo Guerranti, Zinuo Yi, Anna Starovoit, Rafiq Kamel, Simon Geisler, Stephan Günnemann

**Abstract**: Contrastive learning (CL) has emerged as a powerful framework for learning representations of images and text in a self-supervised manner while enhancing model robustness against adversarial attacks. More recently, researchers have extended the principles of contrastive learning to graph-structured data, giving birth to the field of graph contrastive learning (GCL). However, whether GCL methods can deliver the same advantages in adversarial robustness as their counterparts in the image and text domains remains an open question. In this paper, we introduce a comprehensive robustness evaluation protocol tailored to assess the robustness of GCL models. We subject these models to adaptive adversarial attacks targeting the graph structure, specifically in the evasion scenario. We evaluate node and graph classification tasks using diverse real-world datasets and attack strategies. With our work, we aim to offer insights into the robustness of GCL methods and hope to open avenues for potential future research directions.



## **47. Adversarial Attacks and Defenses for Wireless Signal Classifiers using CDI-aware GANs**

cs.IT

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.18820v1) [paper-pdf](http://arxiv.org/pdf/2311.18820v1)

**Authors**: Sujata Sinha, Alkan Soysal

**Abstract**: We introduce a Channel Distribution Information (CDI)-aware Generative Adversarial Network (GAN), designed to address the unique challenges of adversarial attacks in wireless communication systems. The generator in this CDI-aware GAN maps random input noise to the feature space, generating perturbations intended to deceive a target modulation classifier. Its discriminators play a dual role: one enforces that the perturbations follow a Gaussian distribution, making them indistinguishable from Gaussian noise, while the other ensures these perturbations account for realistic channel effects and resemble no-channel perturbations.   Our proposed CDI-aware GAN can be used as an attacker and a defender. In attack scenarios, the CDI-aware GAN demonstrates its prowess by generating robust adversarial perturbations that effectively deceive the target classifier, outperforming known methods. Furthermore, CDI-aware GAN as a defender significantly improves the target classifier's resilience against adversarial attacks.



## **48. Improving the Robustness of Quantized Deep Neural Networks to White-Box Attacks using Stochastic Quantization and Information-Theoretic Ensemble Training**

cs.CV

9 pages, 9 figures, 4 tables

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00105v1) [paper-pdf](http://arxiv.org/pdf/2312.00105v1)

**Authors**: Saurabh Farkya, Aswin Raghavan, Avi Ziskind

**Abstract**: Most real-world applications that employ deep neural networks (DNNs) quantize them to low precision to reduce the compute needs. We present a method to improve the robustness of quantized DNNs to white-box adversarial attacks. We first tackle the limitation of deterministic quantization to fixed ``bins'' by introducing a differentiable Stochastic Quantizer (SQ). We explore the hypothesis that different quantizations may collectively be more robust than each quantized DNN. We formulate a training objective to encourage different quantized DNNs to learn different representations of the input image. The training objective captures diversity and accuracy via mutual information between ensemble members. Through experimentation, we demonstrate substantial improvement in robustness against $L_\infty$ attacks even if the attacker is allowed to backpropagate through SQ (e.g., > 50\% accuracy to PGD(5/255) on CIFAR10 without adversarial training), compared to vanilla DNNs as well as existing ensembles of quantized DNNs. We extend the method to detect attacks and generate robustness profiles in the adversarial information plane (AIP), towards a unified analysis of different threat models by correlating the MI and accuracy.



## **49. Differentiable JPEG: The Devil is in the Details**

cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2309.06978v3) [paper-pdf](http://arxiv.org/pdf/2309.06978v3)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.



## **50. Diffusion Models for Imperceptible and Transferable Adversarial Attack**

cs.CV

Code Page: https://github.com/WindVChen/DiffAttack. In Paper Version  v2, we incorporate more discussions and experiments

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2305.08192v2) [paper-pdf](http://arxiv.org/pdf/2305.08192v2)

**Authors**: Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi

**Abstract**: Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models. Specifically, instead of direct manipulation in pixel space, we craft perturbations in the latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an implicit recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, DiffAttack, is the first that introduces diffusion models into the adversarial attack field. Extensive experiments on various model structures, datasets, and defense methods have demonstrated the superiority of our attack over the existing attack methods.



