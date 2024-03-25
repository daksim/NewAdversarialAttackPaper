# Latest Adversarial Attack Papers
**update at 2024-03-25 09:34:51**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. From Hardware Fingerprint to Access Token: Enhancing the Authentication on IoT Devices**

cs.CR

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15271v1) [paper-pdf](http://arxiv.org/pdf/2403.15271v1)

**Authors**: Yue Xiao, Yi He, Xiaoli Zhang, Qian Wang, Renjie Xie, Kun Sun, Ke Xu, Qi Li

**Abstract**: The proliferation of consumer IoT products in our daily lives has raised the need for secure device authentication and access control. Unfortunately, these resource-constrained devices typically use token-based authentication, which is vulnerable to token compromise attacks that allow attackers to impersonate the devices and perform malicious operations by stealing the access token. Using hardware fingerprints to secure their authentication is a promising way to mitigate these threats. However, once attackers have stolen some hardware fingerprints (e.g., via MitM attacks), they can bypass the hardware authentication by training a machine learning model to mimic fingerprints or reusing these fingerprints to craft forge requests.   In this paper, we present MCU-Token, a secure hardware fingerprinting framework for MCU-based IoT devices even if the cryptographic mechanisms (e.g., private keys) are compromised. MCU-Token can be easily integrated with various IoT devices by simply adding a short hardware fingerprint-based token to the existing payload. To prevent the reuse of this token, we propose a message mapping approach that binds the token to a specific request via generating the hardware fingerprints based on the request payload. To defeat the machine learning attacks, we mix the valid fingerprints with poisoning data so that attackers cannot train a usable model with the leaked tokens. MCU-Token can defend against armored adversary who may replay, craft, and offload the requests via MitM or use both hardware (e.g., use identical devices) and software (e.g., machine learning attacks) strategies to mimic the fingerprints. The system evaluation shows that MCU-Token can achieve high accuracy (over 97%) with a low overhead across various IoT devices and application scenarios.



## **2. Robust optimization for adversarial learning with finite sample complexity guarantees**

cs.LG

**SubmitDate**: 2024-03-22    [abs](http://arxiv.org/abs/2403.15207v1) [paper-pdf](http://arxiv.org/pdf/2403.15207v1)

**Authors**: André Bertolace, Konstatinos Gatsis, Kostas Margellos

**Abstract**: Decision making and learning in the presence of uncertainty has attracted significant attention in view of the increasing need to achieve robust and reliable operations. In the case where uncertainty stems from the presence of adversarial attacks this need is becoming more prominent. In this paper we focus on linear and nonlinear classification problems and propose a novel adversarial training method for robust classifiers, inspired by Support Vector Machine (SVM) margins. We view robustness under a data driven lens, and derive finite sample complexity bounds for both linear and non-linear classifiers in binary and multi-class scenarios. Notably, our bounds match natural classifiers' complexity. Our algorithm minimizes a worst-case surrogate loss using Linear Programming (LP) and Second Order Cone Programming (SOCP) for linear and non-linear models. Numerical experiments on the benchmark MNIST and CIFAR10 datasets show our approach's comparable performance to state-of-the-art methods, without needing adversarial examples during training. Our work offers a comprehensive framework for enhancing binary linear and non-linear classifier robustness, embedding robustness in learning under the presence of adversaries.



## **3. TTPXHunter: Actionable Threat Intelligence Extraction as TTPs from Finished Cyber Threat Reports**

cs.CR

Under Review

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.03267v3) [paper-pdf](http://arxiv.org/pdf/2403.03267v3)

**Authors**: Nanda Rani, Bikash Saha, Vikas Maurya, Sandeep Kumar Shukla

**Abstract**: Understanding the modus operandi of adversaries aids organizations in employing efficient defensive strategies and sharing intelligence in the community. This knowledge is often present in unstructured natural language text within threat analysis reports. A translation tool is needed to interpret the modus operandi explained in the sentences of the threat report and translate it into a structured format. This research introduces a methodology named TTPXHunter for the automated extraction of threat intelligence in terms of Tactics, Techniques, and Procedures (TTPs) from finished cyber threat reports. It leverages cyber domain-specific state-of-the-art natural language processing (NLP) to augment sentences for minority class TTPs and refine pinpointing the TTPs in threat analysis reports significantly. The knowledge of threat intelligence in terms of TTPs is essential for comprehensively understanding cyber threats and enhancing detection and mitigation strategies. We create two datasets: an augmented sentence-TTP dataset of 39,296 samples and a 149 real-world cyber threat intelligence report-to-TTP dataset. Further, we evaluate TTPXHunter on the augmented sentence dataset and the cyber threat reports. The TTPXHunter achieves the highest performance of 92.42% f1-score on the augmented dataset, and it also outperforms existing state-of-the-art solutions in TTP extraction by achieving an f1-score of 97.09% when evaluated over the report dataset. TTPXHunter significantly improves cybersecurity threat intelligence by offering quick, actionable insights into attacker behaviors. This advancement automates threat intelligence analysis, providing a crucial tool for cybersecurity professionals fighting cyber threats.



## **4. Diffusion Attack: Leveraging Stable Diffusion for Naturalistic Image Attacking**

cs.CV

Accepted to IEEE VRW

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14778v1) [paper-pdf](http://arxiv.org/pdf/2403.14778v1)

**Authors**: Qianyu Guo, Jiaming Fu, Yawen Lu, Dongming Gan

**Abstract**: In Virtual Reality (VR), adversarial attack remains a significant security threat. Most deep learning-based methods for physical and digital adversarial attacks focus on enhancing attack performance by crafting adversarial examples that contain large printable distortions that are easy for human observers to identify. However, attackers rarely impose limitations on the naturalness and comfort of the appearance of the generated attack image, resulting in a noticeable and unnatural attack. To address this challenge, we propose a framework to incorporate style transfer to craft adversarial inputs of natural styles that exhibit minimal detectability and maximum natural appearance, while maintaining superior attack capabilities.



## **5. Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures**

cs.CV

32 pages, 15 Tables, and 9 Figures

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14772v1) [paper-pdf](http://arxiv.org/pdf/2403.14772v1)

**Authors**: Sayanton V. Dibbo, Adam Breuer, Juston Moore, Michael Teti

**Abstract**: Recent model inversion attack algorithms permit adversaries to reconstruct a neural network's private training data just by repeatedly querying the network and inspecting its outputs. In this work, we develop a novel network architecture that leverages sparse-coding layers to obtain superior robustness to this class of attacks. Three decades of computer science research has studied sparse coding in the context of image denoising, object recognition, and adversarial misclassification settings, but to the best of our knowledge, its connection to state-of-the-art privacy vulnerabilities remains unstudied. However, sparse coding architectures suggest an advantageous means to defend against model inversion attacks because they allow us to control the amount of irrelevant private information encoded in a network's intermediate representations in a manner that can be computed efficiently during training and that is known to have little effect on classification accuracy. Specifically, compared to networks trained with a variety of state-of-the-art defenses, our sparse-coding architectures maintain comparable or higher classification accuracy while degrading state-of-the-art training data reconstructions by factors of 1.1 to 18.3 across a variety of reconstruction quality metrics (PSNR, SSIM, FID). This performance advantage holds across 5 datasets ranging from CelebA faces to medical images and CIFAR-10, and across various state-of-the-art SGD-based and GAN-based inversion attacks, including Plug-&-Play attacks. We provide a cluster-ready PyTorch codebase to promote research and standardize defense evaluations.



## **6. TMI! Finetuned Models Leak Private Information from their Pretraining Data**

cs.LG

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2306.01181v2) [paper-pdf](http://arxiv.org/pdf/2306.01181v2)

**Authors**: John Abascal, Stanley Wu, Alina Oprea, Jonathan Ullman

**Abstract**: Transfer learning has become an increasingly popular technique in machine learning as a way to leverage a pretrained model trained for one task to assist with building a finetuned model for a related task. This paradigm has been especially popular for $\textit{privacy}$ in machine learning, where the pretrained model is considered public, and only the data for finetuning is considered sensitive. However, there are reasons to believe that the data used for pretraining is still sensitive, making it essential to understand how much information the finetuned model leaks about the pretraining data. In this work we propose a new membership-inference threat model where the adversary only has access to the finetuned model and would like to infer the membership of the pretraining data. To realize this threat model, we implement a novel metaclassifier-based attack, $\textbf{TMI}$, that leverages the influence of memorized pretraining samples on predictions in the downstream task. We evaluate $\textbf{TMI}$ on both vision and natural language tasks across multiple transfer learning settings, including finetuning with differential privacy. Through our evaluation, we find that $\textbf{TMI}$ can successfully infer membership of pretraining examples using query access to the finetuned model. An open-source implementation of $\textbf{TMI}$ can be found $\href{https://github.com/johnmath/tmi-pets24}{\text{on GitHub}}$.



## **7. Adversary-Robust Graph-Based Learning of WSIs**

cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14489v1) [paper-pdf](http://arxiv.org/pdf/2403.14489v1)

**Authors**: Saba Heidari Gheshlaghi, Milan Aryal, Nasim Yahyasoltani, Masoud Ganji

**Abstract**: Enhancing the robustness of deep learning models against adversarial attacks is crucial, especially in critical domains like healthcare where significant financial interests heighten the risk of such attacks. Whole slide images (WSIs) are high-resolution, digitized versions of tissue samples mounted on glass slides, scanned using sophisticated imaging equipment. The digital analysis of WSIs presents unique challenges due to their gigapixel size and multi-resolution storage format. In this work, we aim at improving the robustness of cancer Gleason grading classification systems against adversarial attacks, addressing challenges at both the image and graph levels. As regards the proposed algorithm, we develop a novel and innovative graph-based model which utilizes GNN to extract features from the graph representation of WSIs. A denoising module, along with a pooling layer is incorporated to manage the impact of adversarial attacks on the WSIs. The process concludes with a transformer module that classifies various grades of prostate cancer based on the processed data. To assess the effectiveness of the proposed method, we conducted a comparative analysis using two scenarios. Initially, we trained and tested the model without the denoiser using WSIs that had not been exposed to any attack. We then introduced a range of attacks at either the image or graph level and processed them through the proposed network. The performance of the model was evaluated in terms of accuracy and kappa scores. The results from this comparison showed a significant improvement in cancer diagnosis accuracy, highlighting the robustness and efficiency of the proposed method in handling adversarial challenges in the context of medical imaging.



## **8. A task of anomaly detection for a smart satellite Internet of things system**

cs.LG

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14738v1) [paper-pdf](http://arxiv.org/pdf/2403.14738v1)

**Authors**: Zilong Shao

**Abstract**: When the equipment is working, real-time collection of environmental sensor data for anomaly detection is one of the key links to prevent industrial process accidents and network attacks and ensure system security. However, under the environment with specific real-time requirements, the anomaly detection for environmental sensors still faces the following difficulties: (1) The complex nonlinear correlation characteristics between environmental sensor data variables lack effective expression methods, and the distribution between the data is difficult to be captured. (2) it is difficult to ensure the real-time monitoring requirements by using complex machine learning models, and the equipment cost is too high. (3) Too little sample data leads to less labeled data in supervised learning. This paper proposes an unsupervised deep learning anomaly detection system. Based on the generative adversarial network and self-attention mechanism, considering the different feature information contained in the local subsequences, it automatically learns the complex linear and nonlinear dependencies between environmental sensor variables, and uses the anomaly score calculation method combining reconstruction error and discrimination error. It can monitor the abnormal points of real sensor data with high real-time performance and can run on the intelligent satellite Internet of things system, which is suitable for the real working environment. Anomaly detection outperforms baseline methods in most cases and has good interpretability, which can be used to prevent industrial accidents and cyber-attacks for monitoring environmental sensors.



## **9. Adversarial Attacks and Defenses in Automated Control Systems: A Comprehensive Benchmark**

cs.LG

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.13502v2) [paper-pdf](http://arxiv.org/pdf/2403.13502v2)

**Authors**: Vitaliy Pozdnyakov, Aleksandr Kovalenko, Ilya Makarov, Mikhail Drobyshevskiy, Kirill Lukyanov

**Abstract**: Integrating machine learning into Automated Control Systems (ACS) enhances decision-making in industrial process management. One of the limitations to the widespread adoption of these technologies in industry is the vulnerability of neural networks to adversarial attacks. This study explores the threats in deploying deep learning models for fault diagnosis in ACS using the Tennessee Eastman Process dataset. By evaluating three neural networks with different architectures, we subject them to six types of adversarial attacks and explore five different defense methods. Our results highlight the strong vulnerability of models to adversarial samples and the varying effectiveness of defense strategies. We also propose a novel protection approach by combining multiple defense methods and demonstrate it's efficacy. This research contributes several insights into securing machine learning within ACS, ensuring robust fault diagnosis in industrial processes.



## **10. Adversary-Augmented Simulation to evaluate client-fairness on HyperLedger Fabric**

cs.CR

10 pages (2 pages of references), 8 figures

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14342v1) [paper-pdf](http://arxiv.org/pdf/2403.14342v1)

**Authors**: Erwan Mahe, Rouwaida Abdallah, Sara Tucci-Piergiovanni, Pierre-Yves Piriou

**Abstract**: This paper presents a novel adversary model specifically tailored to distributed systems, with the aim to asses the security of blockchain technologies. Building upon literature on adversarial assumptions and capabilities, we include classical notions of failure and communication models to classify and bind the use of adversarial actions. We focus on the effect of these actions on properties of distributed protocols. A significant effort of our research is the integration of this model into the Multi-Agent eXperimenter (MAX) framework. This integration enables realistic simulations of adversarial attacks on blockchain systems. In particular, we have simulated attacks violating a form of client-fairness on HyperLedger Fabric.



## **11. Large Language Models for Blockchain Security: A Systematic Literature Review**

cs.CR

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14280v1) [paper-pdf](http://arxiv.org/pdf/2403.14280v1)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.



## **12. FMM-Attack: A Flow-based Multi-modal Adversarial Attack on Video-based LLMs**

cs.CV

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.13507v2) [paper-pdf](http://arxiv.org/pdf/2403.13507v2)

**Authors**: Jinmin Li, Kuofeng Gao, Yang Bai, Jingyun Zhang, Shu-tao Xia, Yisen Wang

**Abstract**: Despite the remarkable performance of video-based large language models (LLMs), their adversarial threat remains unexplored. To fill this gap, we propose the first adversarial attack tailored for video-based LLMs by crafting flow-based multi-modal adversarial perturbations on a small fraction of frames within a video, dubbed FMM-Attack. Extensive experiments show that our attack can effectively induce video-based LLMs to generate incorrect answers when videos are added with imperceptible adversarial perturbations. Intriguingly, our FMM-Attack can also induce garbling in the model output, prompting video-based LLMs to hallucinate. Overall, our observations inspire a further understanding of multi-modal robustness and safety-related feature alignment across different modalities, which is of great importance for various large multi-modal models. Our code is available at https://github.com/THU-Kingmin/FMM-Attack.



## **13. Quantum-activated neural reservoirs on-chip open up large hardware security models for resilient authentication**

cond-mat.dis-nn

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14188v1) [paper-pdf](http://arxiv.org/pdf/2403.14188v1)

**Authors**: Zhao He, Maxim S. Elizarov, Ning Li, Fei Xiang, Andrea Fratalocchi

**Abstract**: Quantum artificial intelligence is a frontier of artificial intelligence research, pioneering quantum AI-powered circuits to address problems beyond the reach of deep learning with classical architectures. This work implements a large-scale quantum-activated recurrent neural network possessing more than 3 trillion hardware nodes/cm$^2$, originating from repeatable atomic-scale nucleation dynamics in an amorphous material integrated on-chip, controlled with 0.07 nW electric power per readout channel. Compared to the best-performing reservoirs currently reported, this implementation increases the scale of the network by two orders of magnitude and reduces the power consumption by six, reaching power efficiencies in the range of the human brain, dissipating 0.2 nW/neuron. When interrogated by a classical input, the chip implements a large-scale hardware security model, enabling dictionary-free authentication secure against statistical inference attacks, including AI's present and future development, even for an adversary with a copy of all the classical components available. Experimental tests report 99.6% reliability, 100% user authentication accuracy, and an ideal 50% key uniqueness. Due to its quantum nature, the chip supports a bit density per feature size area three times higher than the best technology available, with the capacity to store more than $2^{1104}$ keys in a footprint of 1 cm$^2$. Such a quantum-powered platform could help counteract the emerging form of warfare led by the cybercrime industry in breaching authentication to target small to large-scale facilities, from private users to intelligent energy grids.



## **14. Reversible Jump Attack to Textual Classifiers with Modification Reduction**

cs.CR

**SubmitDate**: 2024-03-21    [abs](http://arxiv.org/abs/2403.14731v1) [paper-pdf](http://arxiv.org/pdf/2403.14731v1)

**Authors**: Mingze Ni, Zhensu Sun, Wei Liu

**Abstract**: Recent studies on adversarial examples expose vulnerabilities of natural language processing (NLP) models. Existing techniques for generating adversarial examples are typically driven by deterministic hierarchical rules that are agnostic to the optimal adversarial examples, a strategy that often results in adversarial samples with a suboptimal balance between magnitudes of changes and attack successes. To this end, in this research we propose two algorithms, Reversible Jump Attack (RJA) and Metropolis-Hasting Modification Reduction (MMR), to generate highly effective adversarial examples and to improve the imperceptibility of the examples, respectively. RJA utilizes a novel randomization mechanism to enlarge the search space and efficiently adapts to a number of perturbed words for adversarial examples. With these generated adversarial examples, MMR applies the Metropolis-Hasting sampler to enhance the imperceptibility of adversarial examples. Extensive experiments demonstrate that RJA-MMR outperforms current state-of-the-art methods in attack performance, imperceptibility, fluency and grammar correctness.



## **15. A Signal Injection Attack Against Zero Involvement Pairing and Authentication for the Internet of Things**

cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.14018v1) [paper-pdf](http://arxiv.org/pdf/2403.14018v1)

**Authors**: Isaac Ahlgren, Jack West, Kyuin Lee, George Thiruvathukal, Neil Klingensmith

**Abstract**: Zero Involvement Pairing and Authentication (ZIPA) is a promising technique for autoprovisioning large networks of Internet-of-Things (IoT) devices. In this work, we present the first successful signal injection attack on a ZIPA system. Most existing ZIPA systems assume there is a negligible amount of influence from the unsecured outside space on the secured inside space. In reality, environmental signals do leak from adjacent unsecured spaces and influence the environment of the secured space. Our attack takes advantage of this fact to perform a signal injection attack on the popular Schurmann & Sigg algorithm. The keys generated by the adversary with a signal injection attack at 95 dBA is within the standard error of the legitimate device.



## **16. AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models**

cs.CL

Published as a conference paper at ICLR 2024. Code is available at  https://github.com/SheltonLiu-N/AutoDAN

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2310.04451v2) [paper-pdf](http://arxiv.org/pdf/2310.04451v2)

**Authors**: Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao

**Abstract**: The aligned Large Language Models (LLMs) are powerful language understanding and decision-making tools that are created through extensive alignment with human feedback. However, these large models remain susceptible to jailbreak attacks, where adversaries manipulate prompts to elicit malicious outputs that should not be given by aligned LLMs. Investigating jailbreak prompts can lead us to delve into the limitations of LLMs and further guide us to secure them. Unfortunately, existing jailbreak techniques suffer from either (1) scalability issues, where attacks heavily rely on manual crafting of prompts, or (2) stealthiness problems, as attacks depend on token-based algorithms to generate prompts that are often semantically meaningless, making them susceptible to detection through basic perplexity testing. In light of these challenges, we intend to answer this question: Can we develop an approach that can automatically generate stealthy jailbreak prompts? In this paper, we introduce AutoDAN, a novel jailbreak attack against aligned LLMs. AutoDAN can automatically generate stealthy jailbreak prompts by the carefully designed hierarchical genetic algorithm. Extensive evaluations demonstrate that AutoDAN not only automates the process while preserving semantic meaningfulness, but also demonstrates superior attack strength in cross-model transferability, and cross-sample universality compared with the baseline. Moreover, we also compare AutoDAN with perplexity-based defense methods and show that AutoDAN can bypass them effectively.



## **17. Certified Human Trajectory Prediction**

cs.CV

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13778v1) [paper-pdf](http://arxiv.org/pdf/2403.13778v1)

**Authors**: Mohammadhossein Bahari, Saeed Saadatnejad, Amirhossein Asgari Farsangi, Seyed-Mohsen Moosavi-Dezfooli, Alexandre Alahi

**Abstract**: Trajectory prediction plays an essential role in autonomous vehicles. While numerous strategies have been developed to enhance the robustness of trajectory prediction models, these methods are predominantly heuristic and do not offer guaranteed robustness against adversarial attacks and noisy observations. In this work, we propose a certification approach tailored for the task of trajectory prediction. To this end, we address the inherent challenges associated with trajectory prediction, including unbounded outputs, and mutli-modality, resulting in a model that provides guaranteed robustness. Furthermore, we integrate a denoiser into our method to further improve the performance. Through comprehensive evaluations, we demonstrate the effectiveness of the proposed technique across various baselines and using standard trajectory prediction datasets. The code will be made available online: https://s-attack.github.io/



## **18. Defending Against Indirect Prompt Injection Attacks With Spotlighting**

cs.CR

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.14720v1) [paper-pdf](http://arxiv.org/pdf/2403.14720v1)

**Authors**: Keegan Hines, Gary Lopez, Matthew Hall, Federico Zarfati, Yonatan Zunger, Emre Kiciman

**Abstract**: Large Language Models (LLMs), while powerful, are built and trained to process a single text input. In common applications, multiple inputs can be processed by concatenating them together into a single stream of text. However, the LLM is unable to distinguish which sections of prompt belong to various input sources. Indirect prompt injection attacks take advantage of this vulnerability by embedding adversarial instructions into untrusted data being processed alongside user commands. Often, the LLM will mistake the adversarial instructions as user commands to be followed, creating a security vulnerability in the larger system. We introduce spotlighting, a family of prompt engineering techniques that can be used to improve LLMs' ability to distinguish among multiple sources of input. The key insight is to utilize transformations of an input to provide a reliable and continuous signal of its provenance. We evaluate spotlighting as a defense against indirect prompt injection attacks, and find that it is a robust defense that has minimal detrimental impact to underlying NLP tasks. Using GPT-family models, we find that spotlighting reduces the attack success rate from greater than {50}\% to below {2}\% in our experiments with minimal impact on task efficacy.



## **19. On the Privacy Effect of Data Enhancement via the Lens of Memorization**

cs.LG

Accepted by IEEE TIFS, 17 pages

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2208.08270v3) [paper-pdf](http://arxiv.org/pdf/2208.08270v3)

**Authors**: Xiao Li, Qiongxiu Li, Zhanhao Hu, Xiaolin Hu

**Abstract**: Machine learning poses severe privacy concerns as it has been shown that the learned models can reveal sensitive information about their training data. Many works have investigated the effect of widely adopted data augmentation and adversarial training techniques, termed data enhancement in the paper, on the privacy leakage of machine learning models. Such privacy effects are often measured by membership inference attacks (MIAs), which aim to identify whether a particular example belongs to the training set or not. We propose to investigate privacy from a new perspective called memorization. Through the lens of memorization, we find that previously deployed MIAs produce misleading results as they are less likely to identify samples with higher privacy risks as members compared to samples with low privacy risks. To solve this problem, we deploy a recent attack that can capture individual samples' memorization degrees for evaluation. Through extensive experiments, we unveil several findings about the connections between three essential properties of machine learning models, including privacy, generalization gap, and adversarial robustness. We demonstrate that the generalization gap and privacy leakage are less correlated than those of the previous results. Moreover, there is not necessarily a trade-off between adversarial robustness and privacy as stronger adversarial robustness does not make the model more susceptible to privacy attacks.



## **20. Capsule Neural Networks as Noise Stabilizer for Time Series Data**

cs.LG

3 pages, 3 figures

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13867v1) [paper-pdf](http://arxiv.org/pdf/2403.13867v1)

**Authors**: Soyeon Kim, Jihyeon Seong, Hyunkyung Han, Jaesik Choi

**Abstract**: Capsule Neural Networks utilize capsules, which bind neurons into a single vector and learn position equivariant features, which makes them more robust than original Convolutional Neural Networks. CapsNets employ an affine transformation matrix and dynamic routing with coupling coefficients to learn robustly. In this paper, we investigate the effectiveness of CapsNets in analyzing highly sensitive and noisy time series sensor data. To demonstrate CapsNets robustness, we compare their performance with original CNNs on electrocardiogram data, a medical time series sensor data with complex patterns and noise. Our study provides empirical evidence that CapsNets function as noise stabilizers, as investigated by manual and adversarial attack experiments using the fast gradient sign method and three manual attacks, including offset shifting, gradual drift, and temporal lagging. In summary, CapsNets outperform CNNs in both manual and adversarial attacked data. Our findings suggest that CapsNets can be effectively applied to various sensor systems to improve their resilience to noise attacks. These results have significant implications for designing and implementing robust machine learning models in real world applications. Additionally, this study contributes to the effectiveness of CapsNet models in handling noisy data and highlights their potential for addressing the challenges of noise data in time series analysis.



## **21. Have You Poisoned My Data? Defending Neural Networks against Data Poisoning**

cs.LG

Paper accepted for publication at European Symposium on Research in  Computer Security (ESORICS) 2024

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13523v1) [paper-pdf](http://arxiv.org/pdf/2403.13523v1)

**Authors**: Fabio De Gaspari, Dorjan Hitaj, Luigi V. Mancini

**Abstract**: The unprecedented availability of training data fueled the rapid development of powerful neural networks in recent years. However, the need for such large amounts of data leads to potential threats such as poisoning attacks: adversarial manipulations of the training data aimed at compromising the learned model to achieve a given adversarial goal.   This paper investigates defenses against clean-label poisoning attacks and proposes a novel approach to detect and filter poisoned datapoints in the transfer learning setting. We define a new characteristic vector representation of datapoints and show that it effectively captures the intrinsic properties of the data distribution. Through experimental analysis, we demonstrate that effective poisons can be successfully differentiated from clean points in the characteristic vector space. We thoroughly evaluate our proposed approach and compare it to existing state-of-the-art defenses using multiple architectures, datasets, and poison budgets. Our evaluation shows that our proposal outperforms existing approaches in defense rate and final trained model performance across all experimental settings.



## **22. DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation**

cs.CV

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13322v1) [paper-pdf](http://arxiv.org/pdf/2403.13322v1)

**Authors**: Yifan Wu, Jiawei Du, Ping Liu, Yuewei Lin, Wenqing Cheng, Wei Xu

**Abstract**: Dataset distillation is an advanced technique aimed at compressing datasets into significantly smaller counterparts, while preserving formidable training performance. Significant efforts have been devoted to promote evaluation accuracy under limited compression ratio while overlooked the robustness of distilled dataset. In this work, we introduce a comprehensive benchmark that, to the best of our knowledge, is the most extensive to date for evaluating the adversarial robustness of distilled datasets in a unified way. Our benchmark significantly expands upon prior efforts by incorporating a wider range of dataset distillation methods, including the latest advancements such as TESLA and SRe2L, a diverse array of adversarial attack methods, and evaluations across a broader and more extensive collection of datasets such as ImageNet-1K. Moreover, we assessed the robustness of these distilled datasets against representative adversarial attack algorithms like PGD and AutoAttack, while exploring their resilience from a frequency perspective. We also discovered that incorporating distilled data into the training batches of the original dataset can yield to improvement of robustness.



## **23. Enhancing Security in Multi-Robot Systems through Co-Observation Planning, Reachability Analysis, and Network Flow**

cs.RO

12 pages, 6 figures, submitted to IEEE Transactions on Control of  Network Systems

**SubmitDate**: 2024-03-20    [abs](http://arxiv.org/abs/2403.13266v1) [paper-pdf](http://arxiv.org/pdf/2403.13266v1)

**Authors**: Ziqi Yang, Roberto Tron

**Abstract**: This paper addresses security challenges in multi-robot systems (MRS) where adversaries may compromise robot control, risking unauthorized access to forbidden areas. We propose a novel multi-robot optimal planning algorithm that integrates mutual observations and introduces reachability constraints for enhanced security. This ensures that, even with adversarial movements, compromised robots cannot breach forbidden regions without missing scheduled co-observations. The reachability constraint uses ellipsoidal over-approximation for efficient intersection checking and gradient computation. To enhance system resilience and tackle feasibility challenges, we also introduce sub-teams. These cohesive units replace individual robot assignments along each route, enabling redundant robots to deviate for co-observations across different trajectories, securing multiple sub-teams without requiring modifications. We formulate the cross-trajectory co-observation plan by solving a network flow coverage problem on the checkpoint graph generated from the original unsecured MRS trajectories, providing the same security guarantees against plan-deviation attacks. We demonstrate the effectiveness and robustness of our proposed algorithm, which significantly strengthens the security of multi-robot systems in the face of adversarial threats.



## **24. ADAPT to Robustify Prompt Tuning Vision Transformers**

cs.LG

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.13196v1) [paper-pdf](http://arxiv.org/pdf/2403.13196v1)

**Authors**: Masih Eskandar, Tooba Imtiaz, Zifeng Wang, Jennifer Dy

**Abstract**: The performance of deep models, including Vision Transformers, is known to be vulnerable to adversarial attacks. Many existing defenses against these attacks, such as adversarial training, rely on full-model fine-tuning to induce robustness in the models. These defenses require storing a copy of the entire model, that can have billions of parameters, for each task. At the same time, parameter-efficient prompt tuning is used to adapt large transformer-based models to downstream tasks without the need to save large copies. In this paper, we examine parameter-efficient prompt tuning of Vision Transformers for downstream tasks under the lens of robustness. We show that previous adversarial defense methods, when applied to the prompt tuning paradigm, suffer from gradient obfuscation and are vulnerable to adaptive attacks. We introduce ADAPT, a novel framework for performing adaptive adversarial training in the prompt tuning paradigm. Our method achieves competitive robust accuracy of ~40% w.r.t. SOTA robustness methods using full-model fine-tuning, by tuning only ~1% of the number of parameters.



## **25. The Impact of Adversarial Node Placement in Decentralized Federated Learning Networks**

cs.CR

Accepted to ICC 2024 conference

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2311.07946v4) [paper-pdf](http://arxiv.org/pdf/2311.07946v4)

**Authors**: Adam Piaseczny, Eric Ruzomberka, Rohit Parasnis, Christopher G. Brinton

**Abstract**: As Federated Learning (FL) grows in popularity, new decentralized frameworks are becoming widespread. These frameworks leverage the benefits of decentralized environments to enable fast and energy-efficient inter-device communication. However, this growing popularity also intensifies the need for robust security measures. While existing research has explored various aspects of FL security, the role of adversarial node placement in decentralized networks remains largely unexplored. This paper addresses this gap by analyzing the performance of decentralized FL for various adversarial placement strategies when adversaries can jointly coordinate their placement within a network. We establish two baseline strategies for placing adversarial node: random placement and network centrality-based placement. Building on this foundation, we propose a novel attack algorithm that prioritizes adversarial spread over adversarial centrality by maximizing the average network distance between adversaries. We show that the new attack algorithm significantly impacts key performance metrics such as testing accuracy, outperforming the baseline frameworks by between $9\%$ and $66.5\%$ for the considered setups. Our findings provide valuable insights into the vulnerabilities of decentralized FL systems, setting the stage for future research aimed at developing more secure and robust decentralized FL frameworks.



## **26. Review of Generative AI Methods in Cybersecurity**

cs.CR

40 pages

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.08701v2) [paper-pdf](http://arxiv.org/pdf/2403.08701v2)

**Authors**: Yagmur Yigit, William J Buchanan, Madjid G Tehrani, Leandros Maglaras

**Abstract**: Over the last decade, Artificial Intelligence (AI) has become increasingly popular, especially with the use of chatbots such as ChatGPT, Gemini, and DALL-E. With this rise, large language models (LLMs) and Generative AI (GenAI) have also become more prevalent in everyday use. These advancements strengthen cybersecurity's defensive posture and open up new attack avenues for adversaries as well. This paper provides a comprehensive overview of the current state-of-the-art deployments of GenAI, covering assaults, jailbreaking, and applications of prompt injection and reverse psychology. This paper also provides the various applications of GenAI in cybercrimes, such as automated hacking, phishing emails, social engineering, reverse cryptography, creating attack payloads, and creating malware. GenAI can significantly improve the automation of defensive cyber security processes through strategies such as dataset construction, safe code development, threat intelligence, defensive measures, reporting, and cyberattack detection. In this study, we suggest that future research should focus on developing robust ethical norms and innovative defense mechanisms to address the current issues that GenAI creates and to also further encourage an impartial approach to its future application in cybersecurity. Moreover, we underscore the importance of interdisciplinary approaches further to bridge the gap between scientific developments and ethical considerations.



## **27. As Firm As Their Foundations: Can open-sourced foundation models be used to create adversarial examples for downstream tasks?**

cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12693v1) [paper-pdf](http://arxiv.org/pdf/2403.12693v1)

**Authors**: Anjun Hu, Jindong Gu, Francesco Pinto, Konstantinos Kamnitsas, Philip Torr

**Abstract**: Foundation models pre-trained on web-scale vision-language data, such as CLIP, are widely used as cornerstones of powerful machine learning systems. While pre-training offers clear advantages for downstream learning, it also endows downstream models with shared adversarial vulnerabilities that can be easily identified through the open-sourced foundation model. In this work, we expose such vulnerabilities in CLIP's downstream models and show that foundation models can serve as a basis for attacking their downstream systems. In particular, we propose a simple yet effective adversarial attack strategy termed Patch Representation Misalignment (PRM). Solely based on open-sourced CLIP vision encoders, this method produces adversaries that simultaneously fool more than 20 downstream models spanning 4 common vision-language tasks (semantic segmentation, object detection, image captioning and visual question-answering). Our findings highlight the concerning safety risks introduced by the extensive usage of public foundational models in the development of downstream systems, calling for extra caution in these scenarios.



## **28. RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content**

cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.13031v1) [paper-pdf](http://arxiv.org/pdf/2403.13031v1)

**Authors**: Zhuowen Yuan, Zidi Xiong, Yi Zeng, Ning Yu, Ruoxi Jia, Dawn Song, Bo Li

**Abstract**: Recent advancements in Large Language Models (LLMs) have showcased remarkable capabilities across various tasks in different domains. However, the emergence of biases and the potential for generating harmful content in LLMs, particularly under malicious inputs, pose significant challenges. Current mitigation strategies, while effective, are not resilient under adversarial attacks. This paper introduces Resilient Guardrails for Large Language Models (RigorLLM), a novel framework designed to efficiently and effectively moderate harmful and unsafe inputs and outputs for LLMs. By employing a multi-faceted approach that includes energy-based training data augmentation through Langevin dynamics, optimizing a safe suffix for inputs via minimax optimization, and integrating a fusion-based model combining robust KNN with LLMs based on our data augmentation, RigorLLM offers a robust solution to harmful content moderation. Our experimental evaluations demonstrate that RigorLLM not only outperforms existing baselines like OpenAI API and Perspective API in detecting harmful content but also exhibits unparalleled resilience to jailbreaking attacks. The innovative use of constrained optimization and a fusion-based guardrail approach represents a significant step forward in developing more secure and reliable LLMs, setting a new standard for content moderation frameworks in the face of evolving digital threats.



## **29. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

cs.CR

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12503v1) [paper-pdf](http://arxiv.org/pdf/2403.12503v1)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.



## **30. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12445v1) [paper-pdf](http://arxiv.org/pdf/2403.12445v1)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs). Strengthening adversarial attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can stimulate further research on constructing reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability. In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs. To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods. Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks (e.g., Image-Text Retrieval(ITR), Visual Grounding(VG), Image Captioning(IC)).



## **31. Algorithmic Complexity Attacks on Dynamic Learned Indexes**

cs.DB

VLDB 2024

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12433v1) [paper-pdf](http://arxiv.org/pdf/2403.12433v1)

**Authors**: Rui Yang, Evgenios M. Kornaropoulos, Yue Cheng

**Abstract**: Learned Index Structures (LIS) view a sorted index as a model that learns the data distribution, takes a data element key as input, and outputs the predicted position of the key. The original LIS can only handle lookup operations with no support for updates, rendering it impractical to use for typical workloads. To address this limitation, recent studies have focused on designing efficient dynamic learned indexes. ALEX, as the pioneering dynamic learned index structures, enables dynamism by incorporating a series of design choices, including adaptive key space partitioning, dynamic model retraining, and sophisticated engineering and policies that prioritize read/write performance. While these design choices offer improved average-case performance, the emphasis on flexibility and performance increases the attack surface by allowing adversarial behaviors that maximize ALEX's memory space and time complexity in worst-case scenarios. In this work, we present the first systematic investigation of algorithmic complexity attacks (ACAs) targeting the worst-case scenarios of ALEX. We introduce new ACAs that fall into two categories, space ACAs and time ACAs, which target the memory space and time complexity, respectively. First, our space ACA on data nodes exploits ALEX's gapped array layout and uses Multiple-Choice Knapsack (MCK) to generate an optimal adversarial insertion plan for maximizing the memory consumption at the data node level. Second, our space ACA on internal nodes exploits ALEX's catastrophic cost mitigation mechanism, causing an out-of-memory error with only a few hundred adversarial insertions. Third, our time ACA generates pathological insertions to increase the disparity between the actual key distribution and the linear models of data nodes, deteriorating the runtime performance by up to 1,641X compared to ALEX operating under legitimate workloads.



## **32. Electioneering the Network: Dynamic Multi-Step Adversarial Attacks for Community Canvassing**

cs.LG

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.12399v1) [paper-pdf](http://arxiv.org/pdf/2403.12399v1)

**Authors**: Saurabh Sharma, Ambuj SIngh

**Abstract**: The problem of online social network manipulation for community canvassing is of real concern in today's world. Motivated by the study of voter models, opinion and polarization dynamics on networks, we model community canvassing as a dynamic process over a network enabled via gradient-based attacks on GNNs. Existing attacks on GNNs are all single-step and do not account for the dynamic cascading nature of information diffusion in networks. We consider the realistic scenario where an adversary uses a GNN as a proxy to predict and manipulate voter preferences, especially uncertain voters. Gradient-based attacks on the GNN inform the adversary of strategic manipulations that can be made to proselytize targeted voters. In particular, we explore $\textit{minimum budget attacks for community canvassing}$ (MBACC). We show that the MBACC problem is NP-Hard and propose Dynamic Multi-Step Adversarial Community Canvassing (MAC) to address it. MAC makes dynamic local decisions based on the heuristic of low budget and high second-order influence to convert and perturb target voters. MAC is a dynamic multi-step attack that discovers low-budget and high-influence targets from which efficient cascading attacks can happen. We evaluate MAC against single-step baselines on the MBACC problem with multiple underlying networks and GNN models. Our experiments show the superiority of MAC which is able to discover efficient multi-hop attacks for adversarial community canvassing. Our code implementation and data is available at https://github.com/saurabhsharma1993/mac.



## **33. Securely Fine-tuning Pre-trained Encoders Against Adversarial Examples**

cs.CV

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2403.10801v2) [paper-pdf](http://arxiv.org/pdf/2403.10801v2)

**Authors**: Ziqi Zhou, Minghui Li, Wei Liu, Shengshan Hu, Yechao Zhang, Wei Wan, Lulu Xue, Leo Yu Zhang, Dezhong Yao, Hai Jin

**Abstract**: With the evolution of self-supervised learning, the pre-training paradigm has emerged as a predominant solution within the deep learning landscape. Model providers furnish pre-trained encoders designed to function as versatile feature extractors, enabling downstream users to harness the benefits of expansive models with minimal effort through fine-tuning. Nevertheless, recent works have exposed a vulnerability in pre-trained encoders, highlighting their susceptibility to downstream-agnostic adversarial examples (DAEs) meticulously crafted by attackers. The lingering question pertains to the feasibility of fortifying the robustness of downstream models against DAEs, particularly in scenarios where the pre-trained encoders are publicly accessible to the attackers.   In this paper, we initially delve into existing defensive mechanisms against adversarial examples within the pre-training paradigm. Our findings reveal that the failure of current defenses stems from the domain shift between pre-training data and downstream tasks, as well as the sensitivity of encoder parameters. In response to these challenges, we propose Genetic Evolution-Nurtured Adversarial Fine-tuning (Gen-AF), a two-stage adversarial fine-tuning approach aimed at enhancing the robustness of downstream models. Our extensive experiments, conducted across ten self-supervised training methods and six datasets, demonstrate that Gen-AF attains high testing accuracy and robust testing accuracy against state-of-the-art DAEs.



## **34. Improving Visual Quality and Transferability of Adversarial Attacks on Face Recognition Simultaneously with Adversarial Restoration**

cs.CV

\copyright 2023 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2024-03-19    [abs](http://arxiv.org/abs/2309.01582v4) [paper-pdf](http://arxiv.org/pdf/2309.01582v4)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Ping Li

**Abstract**: Adversarial face examples possess two critical properties: Visual Quality and Transferability. However, existing approaches rarely address these properties simultaneously, leading to subpar results. To address this issue, we propose a novel adversarial attack technique known as Adversarial Restoration (AdvRestore), which enhances both visual quality and transferability of adversarial face examples by leveraging a face restoration prior. In our approach, we initially train a Restoration Latent Diffusion Model (RLDM) designed for face restoration. Subsequently, we employ the inference process of RLDM to generate adversarial face examples. The adversarial perturbations are applied to the intermediate features of RLDM. Additionally, by treating RLDM face restoration as a sibling task, the transferability of the generated adversarial face examples is further improved. Our experimental results validate the effectiveness of the proposed attack method.



## **35. Large language models in 6G security: challenges and opportunities**

cs.CR

29 pages, 2 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.12239v1) [paper-pdf](http://arxiv.org/pdf/2403.12239v1)

**Authors**: Tri Nguyen, Huong Nguyen, Ahmad Ijaz, Saeid Sheikhi, Athanasios V. Vasilakos, Panos Kostakos

**Abstract**: The rapid integration of Generative AI (GenAI) and Large Language Models (LLMs) in sectors such as education and healthcare have marked a significant advancement in technology. However, this growth has also led to a largely unexplored aspect: their security vulnerabilities. As the ecosystem that includes both offline and online models, various tools, browser plugins, and third-party applications continues to expand, it significantly widens the attack surface, thereby escalating the potential for security breaches. These expansions in the 6G and beyond landscape provide new avenues for adversaries to manipulate LLMs for malicious purposes. We focus on the security aspects of LLMs from the viewpoint of potential adversaries. We aim to dissect their objectives and methodologies, providing an in-depth analysis of known security weaknesses. This will include the development of a comprehensive threat taxonomy, categorizing various adversary behaviors. Also, our research will concentrate on how LLMs can be integrated into cybersecurity efforts by defense teams, also known as blue teams. We will explore the potential synergy between LLMs and blockchain technology, and how this combination could lead to the development of next-generation, fully autonomous security solutions. This approach aims to establish a unified cybersecurity strategy across the entire computing continuum, enhancing overall digital security infrastructure.



## **36. Adversarial Training Should Be Cast as a Non-Zero-Sum Game**

cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2306.11035v2) [paper-pdf](http://arxiv.org/pdf/2306.11035v2)

**Authors**: Alexander Robey, Fabian Latorre, George J. Pappas, Hamed Hassani, Volkan Cevher

**Abstract**: One prominent approach toward resolving the adversarial vulnerability of deep neural networks is the two-player zero-sum paradigm of adversarial training, in which predictors are trained against adversarially chosen perturbations of data. Despite the promise of this approach, algorithms based on this paradigm have not engendered sufficient levels of robustness and suffer from pathological behavior like robust overfitting. To understand this shortcoming, we first show that the commonly used surrogate-based relaxation used in adversarial training algorithms voids all guarantees on the robustness of trained classifiers. The identification of this pitfall informs a novel non-zero-sum bilevel formulation of adversarial training, wherein each player optimizes a different objective function. Our formulation yields a simple algorithmic framework that matches and in some cases outperforms state-of-the-art attacks, attains comparable levels of robustness to standard adversarial training algorithms, and does not suffer from robust overfitting.



## **37. Diffusion Denoising as a Certified Defense against Clean-label Poisoning**

cs.CR

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11981v1) [paper-pdf](http://arxiv.org/pdf/2403.11981v1)

**Authors**: Sanghyun Hong, Nicholas Carlini, Alexey Kurakin

**Abstract**: We present a certified defense to clean-label poisoning attacks. These attacks work by injecting a small number of poisoning samples (e.g., 1%) that contain $p$-norm bounded adversarial perturbations into the training data to induce a targeted misclassification of a test-time input. Inspired by the adversarial robustness achieved by $denoised$ $smoothing$, we show how an off-the-shelf diffusion model can sanitize the tampered training data. We extensively test our defense against seven clean-label poisoning attacks and reduce their attack success to 0-16% with only a negligible drop in the test time accuracy. We compare our defense with existing countermeasures against clean-label poisoning, showing that the defense reduces the attack success the most and offers the best model utility. Our results highlight the need for future work on developing stronger clean-label attacks and using our certified yet practical defense as a strong baseline to evaluate these attacks.



## **38. Enhancing the Antidote: Improved Pointwise Certifications against Poisoning Attacks**

cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2308.07553v2) [paper-pdf](http://arxiv.org/pdf/2308.07553v2)

**Authors**: Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: Poisoning attacks can disproportionately influence model behaviour by making small changes to the training corpus. While defences against specific poisoning attacks do exist, they in general do not provide any guarantees, leaving them potentially countered by novel attacks. In contrast, by examining worst-case behaviours Certified Defences make it possible to provide guarantees of the robustness of a sample against adversarial attacks modifying a finite number of training samples, known as pointwise certification. We achieve this by exploiting both Differential Privacy and the Sampled Gaussian Mechanism to ensure the invariance of prediction for each testing instance against finite numbers of poisoned examples. In doing so, our model provides guarantees of adversarial robustness that are more than twice as large as those provided by prior certifications.



## **39. SSCAE -- Semantic, Syntactic, and Context-aware natural language Adversarial Examples generator**

cs.CL

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11833v1) [paper-pdf](http://arxiv.org/pdf/2403.11833v1)

**Authors**: Javad Rafiei Asl, Mohammad H. Rafiei, Manar Alohaly, Daniel Takabi

**Abstract**: Machine learning models are vulnerable to maliciously crafted Adversarial Examples (AEs). Training a machine learning model with AEs improves its robustness and stability against adversarial attacks. It is essential to develop models that produce high-quality AEs. Developing such models has been much slower in natural language processing (NLP) than in areas such as computer vision. This paper introduces a practical and efficient adversarial attack model called SSCAE for \textbf{S}emantic, \textbf{S}yntactic, and \textbf{C}ontext-aware natural language \textbf{AE}s generator. SSCAE identifies important words and uses a masked language model to generate an early set of substitutions. Next, two well-known language models are employed to evaluate the initial set in terms of semantic and syntactic characteristics. We introduce (1) a dynamic threshold to capture more efficient perturbations and (2) a local greedy search to generate high-quality AEs. As a black-box method, SSCAE generates humanly imperceptible and context-aware AEs that preserve semantic consistency and the source language's syntactical and grammatical requirements. The effectiveness and superiority of the proposed SSCAE model are illustrated with fifteen comparative experiments and extensive sensitivity analysis for parameter optimization. SSCAE outperforms the existing models in all experiments while maintaining a higher semantic consistency with a lower query number and a comparable perturbation rate.



## **40. Problem space structural adversarial attacks for Network Intrusion Detection Systems based on Graph Neural Networks**

cs.CR

preprint submitted to IEEE TIFS, under review

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11830v1) [paper-pdf](http://arxiv.org/pdf/2403.11830v1)

**Authors**: Andrea Venturi, Dario Stabili, Mirco Marchetti

**Abstract**: Machine Learning (ML) algorithms have become increasingly popular for supporting Network Intrusion Detection Systems (NIDS). Nevertheless, extensive research has shown their vulnerability to adversarial attacks, which involve subtle perturbations to the inputs of the models aimed at compromising their performance. Recent proposals have effectively leveraged Graph Neural Networks (GNN) to produce predictions based also on the structural patterns exhibited by intrusions to enhance the detection robustness. However, the adoption of GNN-based NIDS introduces new types of risks. In this paper, we propose the first formalization of adversarial attacks specifically tailored for GNN in network intrusion detection. Moreover, we outline and model the problem space constraints that attackers need to consider to carry out feasible structural attacks in real-world scenarios. As a final contribution, we conduct an extensive experimental campaign in which we launch the proposed attacks against state-of-the-art GNN-based NIDS. Our findings demonstrate the increased robustness of the models against classical feature-based adversarial attacks, while highlighting their susceptibility to structure-based attacks.



## **41. Expressive Losses for Verified Robustness via Convex Combinations**

cs.LG

ICLR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2305.13991v3) [paper-pdf](http://arxiv.org/pdf/2305.13991v3)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth, Alessio Lomuscio

**Abstract**: In order to train networks for verified adversarial robustness, it is common to over-approximate the worst-case loss over perturbation regions, resulting in networks that attain verifiability at the expense of standard performance. As shown in recent work, better trade-offs between accuracy and robustness can be obtained by carefully coupling adversarial training with over-approximations. We hypothesize that the expressivity of a loss function, which we formalize as the ability to span a range of trade-offs between lower and upper bounds to the worst-case loss through a single parameter (the over-approximation coefficient), is key to attaining state-of-the-art performance. To support our hypothesis, we show that trivial expressive losses, obtained via convex combinations between adversarial attacks and IBP bounds, yield state-of-the-art results across a variety of settings in spite of their conceptual simplicity. We provide a detailed analysis of the relationship between the over-approximation coefficient and performance profiles across different expressive losses, showing that, while expressivity is essential, better approximations of the worst-case loss are not necessarily linked to superior robustness-accuracy trade-offs.



## **42. Hidden in Plain Sight: Undetectable Adversarial Bias Attacks on Vulnerable Patient Populations**

cs.LG

29 pages, 4 figures

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2402.05713v2) [paper-pdf](http://arxiv.org/pdf/2402.05713v2)

**Authors**: Pranav Kulkarni, Andrew Chan, Nithya Navarathna, Skylar Chan, Paul H. Yi, Vishwa S. Parekh

**Abstract**: The proliferation of artificial intelligence (AI) in radiology has shed light on the risk of deep learning (DL) models exacerbating clinical biases towards vulnerable patient populations. While prior literature has focused on quantifying biases exhibited by trained DL models, demographically targeted adversarial bias attacks on DL models and its implication in the clinical environment remains an underexplored field of research in medical imaging. In this work, we demonstrate that demographically targeted label poisoning attacks can introduce undetectable underdiagnosis bias in DL models. Our results across multiple performance metrics and demographic groups like sex, age, and their intersectional subgroups show that adversarial bias attacks demonstrate high-selectivity for bias in the targeted group by degrading group model performance without impacting overall model performance. Furthermore, our results indicate that adversarial bias attacks result in biased DL models that propagate prediction bias even when evaluated with external datasets.



## **43. Stop Reasoning! When Multimodal LLMs with Chain-of-Thought Reasoning Meets Adversarial Images**

cs.CV

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2402.14899v2) [paper-pdf](http://arxiv.org/pdf/2402.14899v2)

**Authors**: Zefeng Wang, Zhen Han, Shuo Chen, Fan Xue, Zifeng Ding, Xun Xiao, Volker Tresp, Philip Torr, Jindong Gu

**Abstract**: Recently, Multimodal LLMs (MLLMs) have shown a great ability to understand images. However, like traditional vision models, they are still vulnerable to adversarial images. Meanwhile, Chain-of-Thought (CoT) reasoning has been widely explored on MLLMs, which not only improves model's performance, but also enhances model's explainability by giving intermediate reasoning steps. Nevertheless, there is still a lack of study regarding MLLMs' adversarial robustness with CoT and an understanding of what the rationale looks like when MLLMs infer wrong answers with adversarial images. Our research evaluates the adversarial robustness of MLLMs when employing CoT reasoning, finding that CoT marginally improves adversarial robustness against existing attack methods. Moreover, we introduce a novel stop-reasoning attack technique that effectively bypasses the CoT-induced robustness enhancements. Finally, we demonstrate the alterations in CoT reasoning when MLLMs confront adversarial images, shedding light on their reasoning process under adversarial attacks.



## **44. LocalStyleFool: Regional Video Style Transfer Attack Using Segment Anything Model**

cs.CV

Accepted to 2024 IEEE Security and Privacy Workshops (SPW)

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11656v1) [paper-pdf](http://arxiv.org/pdf/2403.11656v1)

**Authors**: Yuxin Cao, Jinghao Li, Xi Xiao, Derui Wang, Minhui Xue, Hao Ge, Wei Liu, Guangwu Hu

**Abstract**: Previous work has shown that well-crafted adversarial perturbations can threaten the security of video recognition systems. Attackers can invade such models with a low query budget when the perturbations are semantic-invariant, such as StyleFool. Despite the query efficiency, the naturalness of the minutia areas still requires amelioration, since StyleFool leverages style transfer to all pixels in each frame. To close the gap, we propose LocalStyleFool, an improved black-box video adversarial attack that superimposes regional style-transfer-based perturbations on videos. Benefiting from the popularity and scalably usability of Segment Anything Model (SAM), we first extract different regions according to semantic information and then track them through the video stream to maintain the temporal consistency. Then, we add style-transfer-based perturbations to several regions selected based on the associative criterion of transfer-based gradient information and regional area. Perturbation fine adjustment is followed to make stylized videos adversarial. We demonstrate that LocalStyleFool can improve both intra-frame and inter-frame naturalness through a human-assessed survey, while maintaining competitive fooling rate and query efficiency. Successful experiments on the high-resolution dataset also showcase that scrupulous segmentation of SAM helps to improve the scalability of adversarial attacks under high-resolution data.



## **45. Zeroth-Order Hard-Thresholding: Gradient Error vs. Expansivity**

cs.LG

Accepted for publication at NeurIPS 2022

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2210.05279v2) [paper-pdf](http://arxiv.org/pdf/2210.05279v2)

**Authors**: William de Vazelhes, Hualin Zhang, Huimin Wu, Xiao-Tong Yuan, Bin Gu

**Abstract**: $\ell_0$ constrained optimization is prevalent in machine learning, particularly for high-dimensional problems, because it is a fundamental approach to achieve sparse learning. Hard-thresholding gradient descent is a dominant technique to solve this problem. However, first-order gradients of the objective function may be either unavailable or expensive to calculate in a lot of real-world problems, where zeroth-order (ZO) gradients could be a good surrogate. Unfortunately, whether ZO gradients can work with the hard-thresholding operator is still an unsolved problem. To solve this puzzle, in this paper, we focus on the $\ell_0$ constrained black-box stochastic optimization problems, and propose a new stochastic zeroth-order gradient hard-thresholding (SZOHT) algorithm with a general ZO gradient estimator powered by a novel random support sampling. We provide the convergence analysis of SZOHT under standard assumptions. Importantly, we reveal a conflict between the deviation of ZO estimators and the expansivity of the hard-thresholding operator, and provide a theoretical minimal value of the number of random directions in ZO gradients. In addition, we find that the query complexity of SZOHT is independent or weakly dependent on the dimensionality under different settings. Finally, we illustrate the utility of our method on a portfolio optimization problem as well as black-box adversarial attacks.



## **46. The Lipschitz-Variance-Margin Tradeoff for Enhanced Randomized Smoothing**

cs.LG

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2309.16883v4) [paper-pdf](http://arxiv.org/pdf/2309.16883v4)

**Authors**: Blaise Delattre, Alexandre Araujo, Quentin Barthélemy, Alexandre Allauzen

**Abstract**: Real-life applications of deep neural networks are hindered by their unsteady predictions when faced with noisy inputs and adversarial attacks. The certified radius in this context is a crucial indicator of the robustness of models. However how to design an efficient classifier with an associated certified radius? Randomized smoothing provides a promising framework by relying on noise injection into the inputs to obtain a smoothed and robust classifier. In this paper, we first show that the variance introduced by the Monte-Carlo sampling in the randomized smoothing procedure estimate closely interacts with two other important properties of the classifier, \textit{i.e.} its Lipschitz constant and margin. More precisely, our work emphasizes the dual impact of the Lipschitz constant of the base classifier, on both the smoothed classifier and the empirical variance. To increase the certified robust radius, we introduce a different way to convert logits to probability vectors for the base classifier to leverage the variance-margin trade-off. We leverage the use of Bernstein's concentration inequality along with enhanced Lipschitz bounds for randomized smoothing. Experimental results show a significant improvement in certified accuracy compared to current state-of-the-art methods. Our novel certification procedure allows us to use pre-trained models with randomized smoothing, effectively improving the current certification radius in a zero-shot manner.



## **47. SSAP: A Shape-Sensitive Adversarial Patch for Comprehensive Disruption of Monocular Depth Estimation in Autonomous Navigation Applications**

cs.CV

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11515v1) [paper-pdf](http://arxiv.org/pdf/2403.11515v1)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Ihsen Alouani, Bassem Ouni, Muhammad Shafique

**Abstract**: Monocular depth estimation (MDE) has advanced significantly, primarily through the integration of convolutional neural networks (CNNs) and more recently, Transformers. However, concerns about their susceptibility to adversarial attacks have emerged, especially in safety-critical domains like autonomous driving and robotic navigation. Existing approaches for assessing CNN-based depth prediction methods have fallen short in inducing comprehensive disruptions to the vision system, often limited to specific local areas. In this paper, we introduce SSAP (Shape-Sensitive Adversarial Patch), a novel approach designed to comprehensively disrupt monocular depth estimation (MDE) in autonomous navigation applications. Our patch is crafted to selectively undermine MDE in two distinct ways: by distorting estimated distances or by creating the illusion of an object disappearing from the system's perspective. Notably, our patch is shape-sensitive, meaning it considers the specific shape and scale of the target object, thereby extending its influence beyond immediate proximity. Furthermore, our patch is trained to effectively address different scales and distances from the camera. Experimental results demonstrate that our approach induces a mean depth estimation error surpassing 0.5, impacting up to 99% of the targeted region for CNN-based MDE models. Additionally, we investigate the vulnerability of Transformer-based MDE models to patch-based attacks, revealing that SSAP yields a significant error of 0.59 and exerts substantial influence over 99% of the target region on these models.



## **48. Robust Overfitting Does Matter: Test-Time Adversarial Purification With FGSM**

cs.CV

CVPR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11448v1) [paper-pdf](http://arxiv.org/pdf/2403.11448v1)

**Authors**: Linyu Tang, Lei Zhang

**Abstract**: Numerous studies have demonstrated the susceptibility of deep neural networks (DNNs) to subtle adversarial perturbations, prompting the development of many advanced adversarial defense methods aimed at mitigating adversarial attacks. Current defense strategies usually train DNNs for a specific adversarial attack method and can achieve good robustness in defense against this type of adversarial attack. Nevertheless, when subjected to evaluations involving unfamiliar attack modalities, empirical evidence reveals a pronounced deterioration in the robustness of DNNs. Meanwhile, there is a trade-off between the classification accuracy of clean examples and adversarial examples. Most defense methods often sacrifice the accuracy of clean examples in order to improve the adversarial robustness of DNNs. To alleviate these problems and enhance the overall robust generalization of DNNs, we propose the Test-Time Pixel-Level Adversarial Purification (TPAP) method. This approach is based on the robust overfitting characteristic of DNNs to the fast gradient sign method (FGSM) on training and test datasets. It utilizes FGSM for adversarial purification, to process images for purifying unknown adversarial perturbations from pixels at testing time in a "counter changes with changelessness" manner, thereby enhancing the defense capability of DNNs against various unknown adversarial attacks. Extensive experimental results show that our method can effectively improve both overall robust generalization of DNNs, notably over previous methods.



## **49. Defense Against Adversarial Attacks on No-Reference Image Quality Models with Gradient Norm Regularization**

cs.CV

accepted by CVPR 2024

**SubmitDate**: 2024-03-18    [abs](http://arxiv.org/abs/2403.11397v1) [paper-pdf](http://arxiv.org/pdf/2403.11397v1)

**Authors**: Yujia Liu, Chenxi Yang, Dingquan Li, Jianhao Ding, Tingting Jiang

**Abstract**: The task of No-Reference Image Quality Assessment (NR-IQA) is to estimate the quality score of an input image without additional information. NR-IQA models play a crucial role in the media industry, aiding in performance evaluation and optimization guidance. However, these models are found to be vulnerable to adversarial attacks, which introduce imperceptible perturbations to input images, resulting in significant changes in predicted scores. In this paper, we propose a defense method to improve the stability in predicted scores when attacked by small perturbations, thus enhancing the adversarial robustness of NR-IQA models. To be specific, we present theoretical evidence showing that the magnitude of score changes is related to the $\ell_1$ norm of the model's gradient with respect to the input image. Building upon this theoretical foundation, we propose a norm regularization training strategy aimed at reducing the $\ell_1$ norm of the gradient, thereby boosting the robustness of NR-IQA models. Experiments conducted on four NR-IQA baseline models demonstrate the effectiveness of our strategy in reducing score changes in the presence of adversarial attacks. To the best of our knowledge, this work marks the first attempt to defend against adversarial attacks on NR-IQA models. Our study offers valuable insights into the adversarial robustness of NR-IQA models and provides a foundation for future research in this area.



## **50. A Modified Word Saliency-Based Adversarial Attack on Text Classification Models**

cs.CL

The paper is a preprint of a version submitted in ICCIDA 2024. It  consists of 10 pages and contains 7 tables

**SubmitDate**: 2024-03-17    [abs](http://arxiv.org/abs/2403.11297v1) [paper-pdf](http://arxiv.org/pdf/2403.11297v1)

**Authors**: Hetvi Waghela, Sneha Rakshit, Jaydip Sen

**Abstract**: This paper introduces a novel adversarial attack method targeting text classification models, termed the Modified Word Saliency-based Adversarial At-tack (MWSAA). The technique builds upon the concept of word saliency to strategically perturb input texts, aiming to mislead classification models while preserving semantic coherence. By refining the traditional adversarial attack approach, MWSAA significantly enhances its efficacy in evading detection by classification systems. The methodology involves first identifying salient words in the input text through a saliency estimation process, which prioritizes words most influential to the model's decision-making process. Subsequently, these salient words are subjected to carefully crafted modifications, guided by semantic similarity metrics to ensure that the altered text remains coherent and retains its original meaning. Empirical evaluations conducted on diverse text classification datasets demonstrate the effectiveness of the proposed method in generating adversarial examples capable of successfully deceiving state-of-the-art classification models. Comparative analyses with existing adversarial attack techniques further indicate the superiority of the proposed approach in terms of both attack success rate and preservation of text coherence.



