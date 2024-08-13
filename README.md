# Latest Adversarial Attack Papers
**update at 2024-08-13 18:58:44**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. On Effects of Steering Latent Representation for Large Language Model Unlearning**

cs.CL

15 pages, 5 figures, 8 tables

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06223v1) [paper-pdf](http://arxiv.org/pdf/2408.06223v1)

**Authors**: Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, Naoya Inoue

**Abstract**: Representation Misdirection for Unlearning (RMU), which steers model representation in the intermediate layer to a target random representation, is an effective method for large language model (LLM) unlearning. Despite its high performance, the underlying cause and explanation remain underexplored. In this paper, we first theoretically demonstrate that steering forget representations in the intermediate layer reduces token confidence, causing LLMs to generate wrong or nonsense responses. Second, we investigate how the coefficient influences the alignment of forget-sample representations with the random direction and hint at the optimal coefficient values for effective unlearning across different network layers. Third, we show that RMU unlearned models are robust against adversarial jailbreak attacks. Last, our empirical analysis shows that RMU is less effective when applied to the middle and later layers in LLMs. To resolve this drawback, we propose Adaptive RMU -- a simple yet effective alternative method that makes unlearning effective with most layers. Extensive experiments demonstrate that Adaptive RMU significantly improves the unlearning performance compared to prior art while incurring no additional computational cost.



## **2. Lancelot: Towards Efficient and Privacy-Preserving Byzantine-Robust Federated Learning within Fully Homomorphic Encryption**

cs.CR

26 pages

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06197v1) [paper-pdf](http://arxiv.org/pdf/2408.06197v1)

**Authors**: Siyang Jiang, Hao Yang, Qipeng Xie, Chuan Ma, Sen Wang, Guoliang Xing

**Abstract**: In sectors such as finance and healthcare, where data governance is subject to rigorous regulatory requirements, the exchange and utilization of data are particularly challenging. Federated Learning (FL) has risen as a pioneering distributed machine learning paradigm that enables collaborative model training across multiple institutions while maintaining data decentralization. Despite its advantages, FL is vulnerable to adversarial threats, particularly poisoning attacks during model aggregation, a process typically managed by a central server. However, in these systems, neural network models still possess the capacity to inadvertently memorize and potentially expose individual training instances. This presents a significant privacy risk, as attackers could reconstruct private data by leveraging the information contained in the model itself. Existing solutions fall short of providing a viable, privacy-preserving BRFL system that is both completely secure against information leakage and computationally efficient. To address these concerns, we propose Lancelot, an innovative and computationally efficient BRFL framework that employs fully homomorphic encryption (FHE) to safeguard against malicious client activities while preserving data privacy. Our extensive testing, which includes medical imaging diagnostics and widely-used public image datasets, demonstrates that Lancelot significantly outperforms existing methods, offering more than a twenty-fold increase in processing speed, all while maintaining data privacy.



## **3. Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment**

cs.CV

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.06079v1) [paper-pdf](http://arxiv.org/pdf/2408.06079v1)

**Authors**: Kejia Zhang, Juanjuan Weng, Zhiming Luo, Shaozi Li

**Abstract**: Despite the significant advances that deep neural networks (DNNs) have achieved in various visual tasks, they still exhibit vulnerability to adversarial examples, leading to serious security concerns. Recent adversarial training techniques have utilized inverse adversarial attacks to generate high-confidence examples, aiming to align the distributions of adversarial examples with the high-confidence regions of their corresponding classes. However, in this paper, our investigation reveals that high-confidence outputs under inverse adversarial attacks are correlated with biased feature activation. Specifically, training with inverse adversarial examples causes the model's attention to shift towards background features, introducing a spurious correlation bias. To address this bias, we propose Debiased High-Confidence Adversarial Training (DHAT), a novel approach that not only aligns the logits of adversarial examples with debiased high-confidence logits obtained from inverse adversarial examples, but also restores the model's attention to its normal state by enhancing foreground logit orthogonality. Extensive experiments demonstrate that DHAT achieves state-of-the-art performance and exhibits robust generalization capabilities across various vision datasets. Additionally, DHAT can seamlessly integrate with existing advanced adversarial training techniques for improving the performance.



## **4. Multimodal Large Language Models for Phishing Webpage Detection and Identification**

cs.CR

To appear in eCrime 2024

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05941v1) [paper-pdf](http://arxiv.org/pdf/2408.05941v1)

**Authors**: Jehyun Lee, Peiyuan Lim, Bryan Hooi, Dinil Mon Divakaran

**Abstract**: To address the challenging problem of detecting phishing webpages, researchers have developed numerous solutions, in particular those based on machine learning (ML) algorithms. Among these, brand-based phishing detection that uses models from Computer Vision to detect if a given webpage is imitating a well-known brand has received widespread attention. However, such models are costly and difficult to maintain, as they need to be retrained with labeled dataset that has to be regularly and continuously collected. Besides, they also need to maintain a good reference list of well-known websites and related meta-data for effective performance.   In this work, we take steps to study the efficacy of large language models (LLMs), in particular the multimodal LLMs, in detecting phishing webpages. Given that the LLMs are pretrained on a large corpus of data, we aim to make use of their understanding of different aspects of a webpage (logo, theme, favicon, etc.) to identify the brand of a given webpage and compare the identified brand with the domain name in the URL to detect a phishing attack. We propose a two-phase system employing LLMs in both phases: the first phase focuses on brand identification, while the second verifies the domain. We carry out comprehensive evaluations on a newly collected dataset. Our experiments show that the LLM-based system achieves a high detection rate at high precision; importantly, it also provides interpretable evidence for the decisions. Our system also performs significantly better than a state-of-the-art brand-based phishing detection system while demonstrating robustness against two known adversarial attacks.



## **5. Classifier Guidance Enhances Diffusion-based Adversarial Purification by Preserving Predictive Information**

cs.CV

Accepted by ECAI 2024

**SubmitDate**: 2024-08-12    [abs](http://arxiv.org/abs/2408.05900v1) [paper-pdf](http://arxiv.org/pdf/2408.05900v1)

**Authors**: Mingkun Zhang, Jianing Li, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Adversarial purification is one of the promising approaches to defend neural networks against adversarial attacks. Recently, methods utilizing diffusion probabilistic models have achieved great success for adversarial purification in image classification tasks. However, such methods fall into the dilemma of balancing the needs for noise removal and information preservation. This paper points out that existing adversarial purification methods based on diffusion models gradually lose sample information during the core denoising process, causing occasional label shift in subsequent classification tasks. As a remedy, we suggest to suppress such information loss by introducing guidance from the classifier confidence. Specifically, we propose Classifier-cOnfidence gUided Purification (COUP) algorithm, which purifies adversarial examples while keeping away from the classifier decision boundary. Experimental results show that COUP can achieve better adversarial robustness under strong attack methods.



## **6. Using Retriever Augmented Large Language Models for Attack Graph Generation**

cs.CR

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05855v1) [paper-pdf](http://arxiv.org/pdf/2408.05855v1)

**Authors**: Renascence Tarafder Prapty, Ashish Kundu, Arun Iyengar

**Abstract**: As the complexity of modern systems increases, so does the importance of assessing their security posture through effective vulnerability management and threat modeling techniques. One powerful tool in the arsenal of cybersecurity professionals is the attack graph, a representation of all potential attack paths within a system that an adversary might exploit to achieve a certain objective. Traditional methods of generating attack graphs involve expert knowledge, manual curation, and computational algorithms that might not cover the entire threat landscape due to the ever-evolving nature of vulnerabilities and exploits. This paper explores the approach of leveraging large language models (LLMs), such as ChatGPT, to automate the generation of attack graphs by intelligently chaining Common Vulnerabilities and Exposures (CVEs) based on their preconditions and effects. It also shows how to utilize LLMs to create attack graphs from threat reports.



## **7. A Diamond Model Analysis on Twitter's Biggest Hack**

cs.CR

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2306.15878v3) [paper-pdf](http://arxiv.org/pdf/2306.15878v3)

**Authors**: Chaitanya Rahalkar

**Abstract**: Cyberattacks have prominently increased over the past few years now, and have targeted actors from a wide variety of domains. Understanding the motivation, infrastructure, attack vectors, etc. behind such attacks is vital to proactively work against preventing such attacks in the future and also to analyze the economic and social impact of such attacks. In this paper, we leverage the diamond model to perform an intrusion analysis case study of the 2020 Twitter account hijacking Cyberattack. We follow this standardized incident response model to map the adversary, capability, infrastructure, and victim and perform a comprehensive analysis of the attack, and the impact posed by the attack from a Cybersecurity policy standpoint.



## **8. Improving Adversarial Transferability with Neighbourhood Gradient Information**

cs.CV

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05745v1) [paper-pdf](http://arxiv.org/pdf/2408.05745v1)

**Authors**: Haijing Guo, Jiafeng Wang, Zhaoyu Chen, Kaixun Jiang, Lingyi Hong, Pinxue Guo, Jinglun Li, Wenqiang Zhang

**Abstract**: Deep neural networks (DNNs) are known to be susceptible to adversarial examples, leading to significant performance degradation. In black-box attack scenarios, a considerable attack performance gap between the surrogate model and the target model persists. This work focuses on enhancing the transferability of adversarial examples to narrow this performance gap. We observe that the gradient information around the clean image, i.e. Neighbourhood Gradient Information, can offer high transferability. Leveraging this, we propose the NGI-Attack, which incorporates Example Backtracking and Multiplex Mask strategies, to use this gradient information and enhance transferability fully. Specifically, we first adopt Example Backtracking to accumulate Neighbourhood Gradient Information as the initial momentum term. Multiplex Mask, which forms a multi-way attack strategy, aims to force the network to focus on non-discriminative regions, which can obtain richer gradient information during only a few iterations. Extensive experiments demonstrate that our approach significantly enhances adversarial transferability. Especially, when attacking numerous defense models, we achieve an average attack success rate of 95.8%. Notably, our method can plugin with any off-the-shelf algorithm to improve their attack performance without additional time cost.



## **9. Graph Agent Network: Empowering Nodes with Decentralized Communications Capabilities for Adversarial Resilience**

cs.LG

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2306.06909v2) [paper-pdf](http://arxiv.org/pdf/2306.06909v2)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Guangquan Xu, Pan Zhou, Wengang Ma, Hanyuan Huang

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.



## **10. StealthDiffusion: Towards Evading Diffusion Forensic Detection through Diffusion Model**

cs.CV

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05669v1) [paper-pdf](http://arxiv.org/pdf/2408.05669v1)

**Authors**: Ziyin Zhou, Ke Sun, Zhongxi Chen, Huafeng Kuang, Xiaoshuai Sun, Rongrong Ji

**Abstract**: The rapid progress in generative models has given rise to the critical task of AI-Generated Content Stealth (AIGC-S), which aims to create AI-generated images that can evade both forensic detectors and human inspection. This task is crucial for understanding the vulnerabilities of existing detection methods and developing more robust techniques. However, current adversarial attacks often introduce visible noise, have poor transferability, and fail to address spectral differences between AI-generated and genuine images. To address this, we propose StealthDiffusion, a framework based on stable diffusion that modifies AI-generated images into high-quality, imperceptible adversarial examples capable of evading state-of-the-art forensic detectors. StealthDiffusion comprises two main components: Latent Adversarial Optimization, which generates adversarial perturbations in the latent space of stable diffusion, and Control-VAE, a module that reduces spectral differences between the generated adversarial images and genuine images without affecting the original diffusion model's generation process. Extensive experiments show that StealthDiffusion is effective in both white-box and black-box settings, transforming AI-generated images into high-quality adversarial forgeries with frequency spectra similar to genuine images. These forgeries are classified as genuine by advanced forensic classifiers and are difficult for humans to distinguish.



## **11. Utilizing Large Language Models to Optimize the Detection and Explainability of Phishing Websites**

cs.CR

**SubmitDate**: 2024-08-11    [abs](http://arxiv.org/abs/2408.05667v1) [paper-pdf](http://arxiv.org/pdf/2408.05667v1)

**Authors**: Sayak Saha Roy, Shirin Nilizadeh

**Abstract**: In this paper, we introduce PhishLang, an open-source, lightweight Large Language Model (LLM) specifically designed for phishing website detection through contextual analysis of the website. Unlike traditional heuristic or machine learning models that rely on static features and struggle to adapt to new threats and deep learning models that are computationally intensive, our model utilizes the advanced language processing capabilities of LLMs to learn granular features that are characteristic of phishing attacks. Furthermore, PhishLang operates with minimal data preprocessing and offers performance comparable to leading deep learning tools, while being significantly faster and less resource-intensive. Over a 3.5-month testing period, PhishLang successfully identified approximately 26K phishing URLs, many of which were undetected by popular antiphishing blocklists, thus demonstrating its potential to aid current detection measures. We also evaluate PhishLang against several realistic adversarial attacks and develop six patches that make it very robust against such threats. Furthermore, we integrate PhishLang with GPT-3.5 Turbo to create \textit{explainable blocklisting} - warnings that provide users with contextual information about different features that led to a website being marked as phishing. Finally, we have open-sourced the PhishLang framework and developed a Chromium-based browser extension and URL scanner website, which implement explainable warnings for end-users.



## **12. Cooperative Abnormal Node Detection with Adversary Resistance**

eess.SY

12 pages, 12 figures

**SubmitDate**: 2024-08-10    [abs](http://arxiv.org/abs/2311.16661v2) [paper-pdf](http://arxiv.org/pdf/2311.16661v2)

**Authors**: Yingying Huangfu, Tian Bai

**Abstract**: This paper presents a novel probabilistic detection scheme called Cooperative Statistical Detection~(CSD) for abnormal node detection while defending against adversarial attacks in cluster-tree wireless sensor networks. The CSD performs a two-phase process: 1) designing a likelihood ratio test~(LRT) for a non-root node at its children from the perspective of packet loss; 2) making an overall decision at the root node based on the aggregated detection data of the nodes over tree branches. In most adversarial scenarios, malicious children knowing the detection policy can generate falsified data to protect the abnormal parent from being detected. To resolve this issue, a mechanism is presented in the CSD to remove untrustworthy information. Through theoretical analysis, we show that the LRT-based method achieves the perfect detection. Furthermore, the optimal removal threshold is derived for falsifications with uncertain strategies and guarantees perfect detection of the CSD. As our simulation results shown, the CSD approach is robust to falsifications and can rapidly reach $99\%$ detection accuracy, even in existing adversarial scenarios, which outperforms the state-of-the-art technology.



## **13. ReToMe-VA: Recursive Token Merging for Video Diffusion-based Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2024-08-10    [abs](http://arxiv.org/abs/2408.05479v1) [paper-pdf](http://arxiv.org/pdf/2408.05479v1)

**Authors**: Ziyi Gao, Kai Chen, Zhipeng Wei, Tingshu Mou, Jingjing Chen, Zhiyu Tan, Hao Li, Yu-Gang Jiang

**Abstract**: Recent diffusion-based unrestricted attacks generate imperceptible adversarial examples with high transferability compared to previous unrestricted attacks and restricted attacks. However, existing works on diffusion-based unrestricted attacks are mostly focused on images yet are seldom explored in videos. In this paper, we propose the Recursive Token Merging for Video Diffusion-based Unrestricted Adversarial Attack (ReToMe-VA), which is the first framework to generate imperceptible adversarial video clips with higher transferability. Specifically, to achieve spatial imperceptibility, ReToMe-VA adopts a Timestep-wise Adversarial Latent Optimization (TALO) strategy that optimizes perturbations in diffusion models' latent space at each denoising step. TALO offers iterative and accurate updates to generate more powerful adversarial frames. TALO can further reduce memory consumption in gradient computation. Moreover, to achieve temporal imperceptibility, ReToMe-VA introduces a Recursive Token Merging (ReToMe) mechanism by matching and merging tokens across video frames in the self-attention module, resulting in temporally consistent adversarial videos. ReToMe concurrently facilitates inter-frame interactions into the attack process, inducing more diverse and robust gradients, thus leading to better adversarial transferability. Extensive experiments demonstrate the efficacy of ReToMe-VA, particularly in surpassing state-of-the-art attacks in adversarial transferability by more than 14.16% on average.



## **14. Towards Resilient and Efficient LLMs: A Comparative Study of Efficiency, Performance, and Adversarial Robustness**

cs.CL

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04585v2) [paper-pdf](http://arxiv.org/pdf/2408.04585v2)

**Authors**: Xiaojing Fan, Chunliang Tao

**Abstract**: With the increasing demand for practical applications of Large Language Models (LLMs), many attention-efficient models have been developed to balance performance and computational cost. However, the adversarial robustness of these models remains under-explored. In this work, we design a framework to investigate the trade-off between efficiency, performance, and adversarial robustness of LLMs by comparing three prominent models with varying levels of complexity and efficiency -- Transformer++, Gated Linear Attention (GLA) Transformer, and MatMul-Free LM -- utilizing the GLUE and AdvGLUE datasets. The AdvGLUE dataset extends the GLUE dataset with adversarial samples designed to challenge model robustness. Our results show that while the GLA Transformer and MatMul-Free LM achieve slightly lower accuracy on GLUE tasks, they demonstrate higher efficiency and either superior or comparative robustness on AdvGLUE tasks compared to Transformer++ across different attack levels. These findings highlight the potential of simplified architectures to achieve a compelling balance between efficiency, performance, and adversarial robustness, offering valuable insights for applications where resource constraints and resilience to adversarial attacks are critical.



## **15. Modeling Electromagnetic Signal Injection Attacks on Camera-based Smart Systems: Applications and Mitigation**

cs.CR

13 pages, 10 figures, 4 tables

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.05124v1) [paper-pdf](http://arxiv.org/pdf/2408.05124v1)

**Authors**: Youqian Zhang, Michael Cheung, Chunxi Yang, Xinwei Zhai, Zitong Shen, Xinyu Ji, Eugene Y. Fu, Sze-Yiu Chau, Xiapu Luo

**Abstract**: Numerous safety- or security-critical systems depend on cameras to perceive their surroundings, further allowing artificial intelligence (AI) to analyze the captured images to make important decisions. However, a concerning attack vector has emerged, namely, electromagnetic waves, which pose a threat to the integrity of these systems. Such attacks enable attackers to manipulate the images remotely, leading to incorrect AI decisions, e.g., autonomous vehicles missing detecting obstacles ahead resulting in collisions. The lack of understanding regarding how different systems react to such attacks poses a significant security risk. Furthermore, no effective solutions have been demonstrated to mitigate this threat.   To address these gaps, we modeled the attacks and developed a simulation method for generating adversarial images. Through rigorous analysis, we confirmed that the effects of the simulated adversarial images are indistinguishable from those from real attacks. This method enables researchers and engineers to rapidly assess the susceptibility of various AI vision applications to these attacks, without the need for constructing complicated attack devices. In our experiments, most of the models demonstrated vulnerabilities to these attacks, emphasizing the need to enhance their robustness. Fortunately, our modeling and simulation method serves as a stepping stone toward developing more resilient models. We present a pilot study on adversarial training to improve their robustness against attacks, and our results demonstrate a significant improvement by recovering up to 91% performance, offering a promising direction for mitigating this threat.



## **16. PriPHiT: Privacy-Preserving Hierarchical Training of Deep Neural Networks**

cs.CV

16 pages, 16 figures, 6 tables

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.05092v1) [paper-pdf](http://arxiv.org/pdf/2408.05092v1)

**Authors**: Yamin Sepehri, Pedram Pad, Pascal Frossard, L. Andrea Dunbar

**Abstract**: The training phase of deep neural networks requires substantial resources and as such is often performed on cloud servers. However, this raises privacy concerns when the training dataset contains sensitive content, e.g., face images. In this work, we propose a method to perform the training phase of a deep learning model on both an edge device and a cloud server that prevents sensitive content being transmitted to the cloud while retaining the desired information. The proposed privacy-preserving method uses adversarial early exits to suppress the sensitive content at the edge and transmits the task-relevant information to the cloud. This approach incorporates noise addition during the training phase to provide a differential privacy guarantee. We extensively test our method on different facial datasets with diverse face attributes using various deep learning architectures, showcasing its outstanding performance. We also demonstrate the effectiveness of privacy preservation through successful defenses against different white-box and deep reconstruction attacks.



## **17. XNN: Paradigm Shift in Mitigating Identity Leakage within Cloud-Enabled Deep Learning**

cs.CR

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04974v1) [paper-pdf](http://arxiv.org/pdf/2408.04974v1)

**Authors**: Kaixin Liu, Huixin Xiong, Bingyu Duan, Zexuan Cheng, Xinyu Zhou, Wanqian Zhang, Xiangyu Zhang

**Abstract**: In the domain of cloud-based deep learning, the imperative for external computational resources coexists with acute privacy concerns, particularly identity leakage. To address this challenge, we introduce XNN and XNN-d, pioneering methodologies that infuse neural network features with randomized perturbations, striking a harmonious balance between utility and privacy. XNN, designed for the training phase, ingeniously blends random permutation with matrix multiplication techniques to obfuscate feature maps, effectively shielding private data from potential breaches without compromising training integrity. Concurrently, XNN-d, devised for the inference phase, employs adversarial training to integrate generative adversarial noise. This technique effectively counters black-box access attacks aimed at identity extraction, while a distilled face recognition network adeptly processes the perturbed features, ensuring accurate identification. Our evaluation demonstrates XNN's effectiveness, significantly outperforming existing methods in reducing identity leakage while maintaining a high model accuracy.



## **18. LiD-FL: Towards List-Decodable Federated Learning**

cs.LG

17 pages, 5 figures

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04963v1) [paper-pdf](http://arxiv.org/pdf/2408.04963v1)

**Authors**: Hong Liu, Liren Shan, Han Bao, Ronghui You, Yuhao Yi, Jiancheng Lv

**Abstract**: Federated learning is often used in environments with many unverified participants. Therefore, federated learning under adversarial attacks receives significant attention. This paper proposes an algorithmic framework for list-decodable federated learning, where a central server maintains a list of models, with at least one guaranteed to perform well. The framework has no strict restriction on the fraction of honest workers, extending the applicability of Byzantine federated learning to the scenario with more than half adversaries. Under proper assumptions on the loss function, we prove a convergence theorem for our method. Experimental results, including image classification tasks with both convex and non-convex losses, demonstrate that the proposed algorithm can withstand the malicious majority under various attacks.



## **19. Adversarially Robust Industrial Anomaly Detection Through Diffusion Model**

cs.LG

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04839v1) [paper-pdf](http://arxiv.org/pdf/2408.04839v1)

**Authors**: Yuanpu Cao, Lu Lin, Jinghui Chen

**Abstract**: Deep learning-based industrial anomaly detection models have achieved remarkably high accuracy on commonly used benchmark datasets. However, the robustness of those models may not be satisfactory due to the existence of adversarial examples, which pose significant threats to the practical deployment of deep anomaly detectors. Recently, it has been shown that diffusion models can be used to purify the adversarial noises and thus build a robust classifier against adversarial attacks. Unfortunately, we found that naively applying this strategy in anomaly detection (i.e., placing a purifier before an anomaly detector) will suffer from a high anomaly miss rate since the purifying process can easily remove both the anomaly signal and the adversarial perturbations, causing the later anomaly detector failed to detect anomalies. To tackle this issue, we explore the possibility of performing anomaly detection and adversarial purification simultaneously. We propose a simple yet effective adversarially robust anomaly detection method, \textit{AdvRAD}, that allows the diffusion model to act both as an anomaly detector and adversarial purifier. We also extend our proposed method for certified robustness to $l_2$ norm bounded perturbations. Through extensive experiments, we show that our proposed method exhibits outstanding (certified) adversarial robustness while also maintaining equally strong anomaly detection performance on par with the state-of-the-art methods on industrial anomaly detection benchmark datasets.



## **20. h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessment**

cs.CR

**SubmitDate**: 2024-08-09    [abs](http://arxiv.org/abs/2408.04811v1) [paper-pdf](http://arxiv.org/pdf/2408.04811v1)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: The safety of Large Language Models (LLMs) remains a critical concern due to a lack of adequate benchmarks for systematically evaluating their ability to resist generating harmful content. Previous efforts towards automated red teaming involve static or templated sets of illicit requests and adversarial prompts which have limited utility given jailbreak attacks' evolving and composable nature. We propose a novel dynamic benchmark of composable jailbreak attacks to move beyond static datasets and taxonomies of attacks and harms. Our approach consists of three components collectively called h4rm3l: (1) a domain-specific language that formally expresses jailbreak attacks as compositions of parameterized prompt transformation primitives, (2) bandit-based few-shot program synthesis algorithms that generate novel attacks optimized to penetrate the safety filters of a target black box LLM, and (3) open-source automated red-teaming software employing the previous two components. We use h4rm3l to generate a dataset of 2656 successful novel jailbreak attacks targeting 6 state-of-the-art (SOTA) open-source and proprietary LLMs. Several of our synthesized attacks are more effective than previously reported ones, with Attack Success Rates exceeding 90% on SOTA closed language models such as claude-3-haiku and GPT4-o. By generating datasets of jailbreak attacks in a unified formal representation, h4rm3l enables reproducible benchmarking and automated red-teaming, contributes to understanding LLM safety limitations, and supports the development of robust defenses in an increasingly LLM-integrated world.   Warning: This paper and related research artifacts contain offensive and potentially disturbing prompts and model-generated content.



## **21. Tamper-Resistant Safeguards for Open-Weight LLMs**

cs.LG

Website: https://www.tamper-resistant-safeguards.com

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.00761v2) [paper-pdf](http://arxiv.org/pdf/2408.00761v2)

**Authors**: Rishub Tamirisa, Bhrugu Bharathi, Long Phan, Andy Zhou, Alice Gatti, Tarun Suresh, Maxwell Lin, Justin Wang, Rowan Wang, Ron Arel, Andy Zou, Dawn Song, Bo Li, Dan Hendrycks, Mantas Mazeika

**Abstract**: Rapid advances in the capabilities of large language models (LLMs) have raised widespread concerns regarding their potential for malicious use. Open-weight LLMs present unique challenges, as existing safeguards lack robustness to tampering attacks that modify model weights. For example, recent works have demonstrated that refusal and unlearning safeguards can be trivially removed with a few steps of fine-tuning. These vulnerabilities necessitate new approaches for enabling the safe release of open-weight LLMs. We develop a method, called TAR, for building tamper-resistant safeguards into open-weight LLMs such that adversaries cannot remove the safeguards even after thousands of steps of fine-tuning. In extensive evaluations and red teaming analyses, we find that our method greatly improves tamper-resistance while preserving benign capabilities. Our results demonstrate that tamper-resistance is a tractable problem, opening up a promising new avenue to improve the safety and security of open-weight LLMs.



## **22. Improving Network Interpretability via Explanation Consistency Evaluation**

cs.CV

To appear in IEEE Transactions on Multimedia

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04600v1) [paper-pdf](http://arxiv.org/pdf/2408.04600v1)

**Authors**: Hefeng Wu, Hao Jiang, Keze Wang, Ziyi Tang, Xianghuan He, Liang Lin

**Abstract**: While deep neural networks have achieved remarkable performance, they tend to lack transparency in prediction. The pursuit of greater interpretability in neural networks often results in a degradation of their original performance. Some works strive to improve both interpretability and performance, but they primarily depend on meticulously imposed conditions. In this paper, we propose a simple yet effective framework that acquires more explainable activation heatmaps and simultaneously increase the model performance, without the need for any extra supervision. Specifically, our concise framework introduces a new metric, i.e., explanation consistency, to reweight the training samples adaptively in model learning. The explanation consistency metric is utilized to measure the similarity between the model's visual explanations of the original samples and those of semantic-preserved adversarial samples, whose background regions are perturbed by using image adversarial attack techniques. Our framework then promotes the model learning by paying closer attention to those training samples with a high difference in explanations (i.e., low explanation consistency), for which the current model cannot provide robust interpretations. Comprehensive experimental results on various benchmarks demonstrate the superiority of our framework in multiple aspects, including higher recognition accuracy, greater data debiasing capability, stronger network robustness, and more precise localization ability on both regular networks and interpretable networks. We also provide extensive ablation studies and qualitative analyses to unveil the detailed contribution of each component.



## **23. Ensemble everything everywhere: Multi-scale aggregation for adversarial robustness**

cs.CV

34 pages, 25 figures, appendix

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.05446v1) [paper-pdf](http://arxiv.org/pdf/2408.05446v1)

**Authors**: Stanislav Fort, Balaji Lakshminarayanan

**Abstract**: Adversarial examples pose a significant challenge to the robustness, reliability and alignment of deep neural networks. We propose a novel, easy-to-use approach to achieving high-quality representations that lead to adversarial robustness through the use of multi-resolution input representations and dynamic self-ensembling of intermediate layer predictions. We demonstrate that intermediate layer predictions exhibit inherent robustness to adversarial attacks crafted to fool the full classifier, and propose a robust aggregation mechanism based on Vickrey auction that we call \textit{CrossMax} to dynamically ensemble them. By combining multi-resolution inputs and robust ensembling, we achieve significant adversarial robustness on CIFAR-10 and CIFAR-100 datasets without any adversarial training or extra data, reaching an adversarial accuracy of $\approx$72% (CIFAR-10) and $\approx$48% (CIFAR-100) on the RobustBench AutoAttack suite ($L_\infty=8/255)$ with a finetuned ImageNet-pretrained ResNet152. This represents a result comparable with the top three models on CIFAR-10 and a +5 % gain compared to the best current dedicated approach on CIFAR-100. Adding simple adversarial training on top, we get $\approx$78% on CIFAR-10 and $\approx$51% on CIFAR-100, improving SOTA by 5 % and 9 % respectively and seeing greater gains on the harder dataset. We validate our approach through extensive experiments and provide insights into the interplay between adversarial robustness, and the hierarchical nature of deep representations. We show that simple gradient-based attacks against our model lead to human-interpretable images of the target classes as well as interpretable image changes. As a byproduct, using our multi-resolution prior, we turn pre-trained classifiers and CLIP models into controllable image generators and develop successful transferable attacks on large vision language models.



## **24. Understanding the Security Benefits and Overheads of Emerging Industry Solutions to DRAM Read Disturbance**

cs.CR

To appear in DRAMSec 2024

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2406.19094v3) [paper-pdf](http://arxiv.org/pdf/2406.19094v3)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Oğuz Ergin, Onur Mutlu

**Abstract**: We present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC), described in JEDEC DDR5 specification's April 2024 update. Unlike prior state-of-the-art that advises the memory controller to periodically issue refresh management (RFM) commands, which provides the DRAM chip with time to perform refreshes, PRAC introduces a new back-off signal. PRAC's back-off signal propagates from the DRAM chip to the memory controller and forces the memory controller to 1) stop serving requests and 2) issue RFM commands. As a result, RFM commands are issued when needed as opposed to periodically, reducing RFM's overheads. We analyze PRAC in four steps. First, we define an adversarial access pattern that represents the worst-case for PRAC's security. Second, we investigate PRAC's configurations and security implications. Our analyses show that PRAC can be configured for secure operation as long as no bitflip occurs before accessing a memory location 10 times. Third, we evaluate the performance impact of PRAC and compare it against prior works using Ramulator 2.0. Our analysis shows that while PRAC incurs less than 13% performance overhead for today's DRAM chips, its performance overheads can reach up to 94% for future DRAM chips that are more vulnerable to read disturbance bitflips. Fourth, we define an availability adversarial access pattern that exacerbates PRAC's performance overhead to perform a memory performance attack, demonstrating that such an adversarial pattern can hog up to 94% of DRAM throughput and degrade system throughput by up to 95%. We discuss PRAC's implications on future systems and foreshadow future research directions. To aid future research, we open-source our implementations and scripts at https://github.com/CMU-SAFARI/ramulator2.



## **25. Constructing Adversarial Examples for Vertical Federated Learning: Optimal Client Corruption through Multi-Armed Bandit**

cs.LG

Published on ICLR2024

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04310v1) [paper-pdf](http://arxiv.org/pdf/2408.04310v1)

**Authors**: Duanyi Yao, Songze Li, Ye Xue, Jin Liu

**Abstract**: Vertical federated learning (VFL), where each participating client holds a subset of data features, has found numerous applications in finance, healthcare, and IoT systems. However, adversarial attacks, particularly through the injection of adversarial examples (AEs), pose serious challenges to the security of VFL models. In this paper, we investigate such vulnerabilities through developing a novel attack to disrupt the VFL inference process, under a practical scenario where the adversary is able to adaptively corrupt a subset of clients. We formulate the problem of finding optimal attack strategies as an online optimization problem, which is decomposed into an inner problem of adversarial example generation (AEG) and an outer problem of corruption pattern selection (CPS). Specifically, we establish the equivalence between the formulated CPS problem and a multi-armed bandit (MAB) problem, and propose the Thompson sampling with Empirical maximum reward (E-TS) algorithm for the adversary to efficiently identify the optimal subset of clients for corruption. The key idea of E-TS is to introduce an estimation of the expected maximum reward for each arm, which helps to specify a small set of competitive arms, on which the exploration for the optimal arm is performed. This significantly reduces the exploration space, which otherwise can quickly become prohibitively large as the number of clients increases. We analytically characterize the regret bound of E-TS, and empirically demonstrate its capability of efficiently revealing the optimal corruption pattern with the highest attack success rate, under various datasets of popular VFL tasks.



## **26. Eliminating Backdoors in Neural Code Models via Trigger Inversion**

cs.CR

Under review

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04683v1) [paper-pdf](http://arxiv.org/pdf/2408.04683v1)

**Authors**: Weisong Sun, Yuchen Chen, Chunrong Fang, Yebo Feng, Yuan Xiao, An Guo, Quanjun Zhang, Yang Liu, Baowen Xu, Zhenyu Chen

**Abstract**: Neural code models (NCMs) have been widely used for addressing various code understanding tasks, such as defect detection and clone detection. However, numerous recent studies reveal that such models are vulnerable to backdoor attacks. Backdoored NCMs function normally on normal code snippets, but exhibit adversary-expected behavior on poisoned code snippets injected with the adversary-crafted trigger. It poses a significant security threat. For example, a backdoored defect detection model may misclassify user-submitted defective code as non-defective. If this insecure code is then integrated into critical systems, like autonomous driving systems, it could lead to life safety. However, there is an urgent need for effective defenses against backdoor attacks targeting NCMs.   To address this issue, in this paper, we innovatively propose a backdoor defense technique based on trigger inversion, called EliBadCode. EliBadCode first filters the model vocabulary for trigger tokens to reduce the search space for trigger inversion, thereby enhancing the efficiency of the trigger inversion. Then, EliBadCode introduces a sample-specific trigger position identification method, which can reduce the interference of adversarial perturbations for subsequent trigger inversion, thereby producing effective inverted triggers efficiently. Subsequently, EliBadCode employs a Greedy Coordinate Gradient algorithm to optimize the inverted trigger and designs a trigger anchoring method to purify the inverted trigger. Finally, EliBadCode eliminates backdoors through model unlearning. We evaluate the effectiveness of EliBadCode in eliminating backdoor attacks against multiple NCMs used for three safety-critical code understanding tasks. The results demonstrate that EliBadCode can effectively eliminate backdoors while having minimal adverse effects on the normal functionality of the model.



## **27. Unveiling Hidden Visual Information: A Reconstruction Attack Against Adversarial Visual Information Hiding**

cs.CV

12 pages

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04261v1) [paper-pdf](http://arxiv.org/pdf/2408.04261v1)

**Authors**: Jonggyu Jang, Hyeonsu Lyu, Seongjin Hwang, Hyun Jong Yang

**Abstract**: This paper investigates the security vulnerabilities of adversarial-example-based image encryption by executing data reconstruction (DR) attacks on encrypted images. A representative image encryption method is the adversarial visual information hiding (AVIH), which uses type-I adversarial example training to protect gallery datasets used in image recognition tasks. In the AVIH method, the type-I adversarial example approach creates images that appear completely different but are still recognized by machines as the original ones. Additionally, the AVIH method can restore encrypted images to their original forms using a predefined private key generative model. For the best security, assigning a unique key to each image is recommended; however, storage limitations may necessitate some images sharing the same key model. This raises a crucial security question for AVIH: How many images can safely share the same key model without being compromised by a DR attack? To address this question, we introduce a dual-strategy DR attack against the AVIH encryption method by incorporating (1) generative-adversarial loss and (2) augmented identity loss, which prevent DR from overfitting -- an issue akin to that in machine learning. Our numerical results validate this approach through image recognition and re-identification benchmarks, demonstrating that our strategy can significantly enhance the quality of reconstructed images, thereby requiring fewer key-sharing encrypted images. Our source code to reproduce our results will be available soon.



## **28. Artificial Intelligence based Approach for Identification and Mitigation of Cyber-Attacks in Wide-Area Control of Power Systems**

eess.SY

AAAI 2023 workshop

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04189v1) [paper-pdf](http://arxiv.org/pdf/2408.04189v1)

**Authors**: Jishnudeep Kar, Aranya Chakrabortty

**Abstract**: We propose a generative adversarial network (GAN) based deep learning method that serves the dual role of both identification and mitigation of cyber-attacks in wide-area damping control loops of power systems. Two specific types of attacks considered are false data injection and denial-of-service (DoS). Unlike existing methods, which are either model-based or model-free and yet require two separate learning modules for detection and mitigation leading to longer response times before clearing an attack, our deep learner incorporate both goals within the same integrated framework. A Long Short-Term Memory (LSTM) encoder-decoder based GAN is proposed that captures the temporal dynamics of the power system significantly more accurately than fully-connected GANs, thereby providing better accuracy and faster response for both goals. The method is validated using the IEEE 68-bus power system model.



## **29. EdgeShield: A Universal and Efficient Edge Computing Framework for Robust AI**

cs.CR

**SubmitDate**: 2024-08-08    [abs](http://arxiv.org/abs/2408.04181v1) [paper-pdf](http://arxiv.org/pdf/2408.04181v1)

**Authors**: Duo Zhong, Bojing Li, Xiang Chen, Chenchen Liu

**Abstract**: The increasing prevalence of adversarial attacks on Artificial Intelligence (AI) systems has created a need for innovative security measures. However, the current methods of defending against these attacks often come with a high computing cost and require back-end processing, making real-time defense challenging. Fortunately, there have been remarkable advancements in edge-computing, which make it easier to deploy neural networks on edge devices. Building upon these advancements, we propose an edge framework design to enable universal and efficient detection of adversarial attacks. This framework incorporates an attention-based adversarial detection methodology and a lightweight detection network formation, making it suitable for a wide range of neural networks and can be deployed on edge devices. To assess the effectiveness of our proposed framework, we conducted evaluations on five neural networks. The results indicate an impressive 97.43% F-score can be achieved, demonstrating the framework's proficiency in detecting adversarial attacks. Moreover, our proposed framework also exhibits significantly reduced computing complexity and cost in comparison to previous detection methods. This aspect is particularly beneficial as it ensures that the defense mechanism can be efficiently implemented in real-time on-edge devices.



## **30. Investigating Adversarial Attacks in Software Analytics via Machine Learning Explainability**

cs.SE

This is paper is under review

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.04124v1) [paper-pdf](http://arxiv.org/pdf/2408.04124v1)

**Authors**: MD Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: With the recent advancements in machine learning (ML), numerous ML-based approaches have been extensively applied in software analytics tasks to streamline software development and maintenance processes. Nevertheless, studies indicate that despite their potential usefulness, ML models are vulnerable to adversarial attacks, which may result in significant monetary losses in these processes. As a result, the ML models' robustness against adversarial attacks must be assessed before they are deployed in software analytics tasks. Despite several techniques being available for adversarial attacks in software analytics tasks, exploring adversarial attacks using ML explainability is largely unexplored. Therefore, this study aims to investigate the relationship between ML explainability and adversarial attacks to measure the robustness of ML models in software analytics tasks. In addition, unlike most existing attacks that directly perturb input-space, our attack approach focuses on perturbing feature-space. Our extensive experiments, involving six datasets, three ML explainability techniques, and seven ML models, demonstrate that ML explainability can be used to conduct successful adversarial attacks on ML models in software analytics tasks. This is achieved by modifying only the top 1-3 important features identified by ML explainability techniques. Consequently, the ML models under attack fail to accurately predict up to 86.6% of instances that were correctly predicted before adversarial attacks, indicating the models' low robustness against such attacks. Finally, our proposed technique demonstrates promising results compared to four state-of-the-art adversarial attack techniques targeting tabular data.



## **31. Effective Prompt Extraction from Language Models**

cs.CL

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2307.06865v3) [paper-pdf](http://arxiv.org/pdf/2307.06865v3)

**Authors**: Yiming Zhang, Nicholas Carlini, Daphne Ippolito

**Abstract**: The text generated by large language models is commonly controlled by prompting, where a prompt prepended to a user's query guides the model's output. The prompts used by companies to guide their models are often treated as secrets, to be hidden from the user making the query. They have even been treated as commodities to be bought and sold on marketplaces. However, anecdotal reports have shown adversarial users employing prompt extraction attacks to recover these prompts. In this paper, we present a framework for systematically measuring the effectiveness of these attacks. In experiments with 3 different sources of prompts and 11 underlying large language models, we find that simple text-based attacks can in fact reveal prompts with high probability. Our framework determines with high precision whether an extracted prompt is the actual secret prompt, rather than a model hallucination. Prompt extraction from real systems such as Claude 3 and ChatGPT further suggest that system prompts can be revealed by an adversary despite existing defenses in place.



## **32. LaFA: Latent Feature Attacks on Non-negative Matrix Factorization**

cs.LG

LA-UR-24-26951

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03909v1) [paper-pdf](http://arxiv.org/pdf/2408.03909v1)

**Authors**: Minh Vu, Ben Nebgen, Erik Skau, Geigh Zollicoffer, Juan Castorena, Kim Rasmussen, Boian Alexandrov, Manish Bhattarai

**Abstract**: As Machine Learning (ML) applications rapidly grow, concerns about adversarial attacks compromising their reliability have gained significant attention. One unsupervised ML method known for its resilience to such attacks is Non-negative Matrix Factorization (NMF), an algorithm that decomposes input data into lower-dimensional latent features. However, the introduction of powerful computational tools such as Pytorch enables the computation of gradients of the latent features with respect to the original data, raising concerns about NMF's reliability. Interestingly, naively deriving the adversarial loss for NMF as in the case of ML would result in the reconstruction loss, which can be shown theoretically to be an ineffective attacking objective. In this work, we introduce a novel class of attacks in NMF termed Latent Feature Attacks (LaFA), which aim to manipulate the latent features produced by the NMF process. Our method utilizes the Feature Error (FE) loss directly on the latent features. By employing FE loss, we generate perturbations in the original data that significantly affect the extracted latent features, revealing vulnerabilities akin to those found in other ML techniques. To handle large peak-memory overhead from gradient back-propagation in FE attacks, we develop a method based on implicit differentiation which enables their scaling to larger datasets. We validate NMF vulnerabilities and FE attacks effectiveness through extensive experiments on synthetic and real-world data.



## **33. Decoding Biases: Automated Methods and LLM Judges for Gender Bias Detection in Language Models**

cs.CL

6 pages paper content, 17 pages of appendix

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03907v1) [paper-pdf](http://arxiv.org/pdf/2408.03907v1)

**Authors**: Shachi H Kumar, Saurav Sahay, Sahisnu Mazumder, Eda Okur, Ramesh Manuvinakurike, Nicole Beckage, Hsuan Su, Hung-yi Lee, Lama Nachman

**Abstract**: Large Language Models (LLMs) have excelled at language understanding and generating human-level text. However, even with supervised training and human alignment, these LLMs are susceptible to adversarial attacks where malicious users can prompt the model to generate undesirable text. LLMs also inherently encode potential biases that can cause various harmful effects during interactions. Bias evaluation metrics lack standards as well as consensus and existing methods often rely on human-generated templates and annotations which are expensive and labor intensive. In this work, we train models to automatically create adversarial prompts to elicit biased responses from target LLMs. We present LLM- based bias evaluation metrics and also analyze several existing automatic evaluation methods and metrics. We analyze the various nuances of model responses, identify the strengths and weaknesses of model families, and assess where evaluation methods fall short. We compare these metrics to human evaluation and validate that the LLM-as-a-Judge metric aligns with human judgement on bias in response generation.



## **34. EnJa: Ensemble Jailbreak on Large Language Models**

cs.CR

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03603v1) [paper-pdf](http://arxiv.org/pdf/2408.03603v1)

**Authors**: Jiahao Zhang, Zilong Wang, Ruofan Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As Large Language Models (LLMs) are increasingly being deployed in safety-critical applications, their vulnerability to potential jailbreaks -- malicious prompts that can disable the safety mechanism of LLMs -- has attracted growing research attention. While alignment methods have been proposed to protect LLMs from jailbreaks, many have found that aligned LLMs can still be jailbroken by carefully crafted malicious prompts, producing content that violates policy regulations. Existing jailbreak attacks on LLMs can be categorized into prompt-level methods which make up stories/logic to circumvent safety alignment and token-level attack methods which leverage gradient methods to find adversarial tokens. In this work, we introduce the concept of Ensemble Jailbreak and explore methods that can integrate prompt-level and token-level jailbreak into a more powerful hybrid jailbreak attack. Specifically, we propose a novel EnJa attack to hide harmful instructions using prompt-level jailbreak, boost the attack success rate using a gradient-based attack, and connect the two types of jailbreak attacks via a template-based connector. We evaluate the effectiveness of EnJa on several aligned models and show that it achieves a state-of-the-art attack success rate with fewer queries and is much stronger than any individual jailbreak.



## **35. Enhancing Output Diversity Improves Conjugate Gradient-based Adversarial Attacks**

cs.LG

ICPRAI2024

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03972v1) [paper-pdf](http://arxiv.org/pdf/2408.03972v1)

**Authors**: Keiichiro Yamamura, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstract**: Deep neural networks are vulnerable to adversarial examples, and adversarial attacks that generate adversarial examples have been studied in this context. Existing studies imply that increasing the diversity of model outputs contributes to improving the attack performance. This study focuses on the Auto Conjugate Gradient (ACG) attack, which is inspired by the conjugate gradient method and has a high diversification performance. We hypothesized that increasing the distance between two consecutive search points would enhance the output diversity. To test our hypothesis, we propose Rescaling-ACG (ReACG), which automatically modifies the two components that significantly affect the distance between two consecutive search points, including the search direction and step size. ReACG showed higher attack performance than that of ACG, and is particularly effective for ImageNet models with several classification classes. Experimental results show that the distance between two consecutive search points enhances the output diversity and may help develop new potent attacks. The code is available at \url{https://github.com/yamamura-k/ReACG}



## **36. Spatial-Frequency Discriminability for Revealing Adversarial Perturbations**

cs.CV

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2305.10856v3) [paper-pdf](http://arxiv.org/pdf/2305.10856v3)

**Authors**: Chao Wang, Shuren Qi, Zhiqiu Huang, Yushu Zhang, Rushi Lan, Xiaochun Cao, Feng-Lei Fan

**Abstract**: The vulnerability of deep neural networks to adversarial perturbations has been widely perceived in the computer vision community. From a security perspective, it poses a critical risk for modern vision systems, e.g., the popular Deep Learning as a Service (DLaaS) frameworks. For protecting deep models while not modifying them, current algorithms typically detect adversarial patterns through discriminative decomposition for natural and adversarial data. However, these decompositions are either biased towards frequency resolution or spatial resolution, thus failing to capture adversarial patterns comprehensively. Also, when the detector relies on few fixed features, it is practical for an adversary to fool the model while evading the detector (i.e., defense-aware attack). Motivated by such facts, we propose a discriminative detector relying on a spatial-frequency Krawtchouk decomposition. It expands the above works from two aspects: 1) the introduced Krawtchouk basis provides better spatial-frequency discriminability, capturing the differences between natural and adversarial data comprehensively in both spatial and frequency distributions, w.r.t. the common trigonometric or wavelet basis; 2) the extensive features formed by the Krawtchouk decomposition allows for adaptive feature selection and secrecy mechanism, significantly increasing the difficulty of the defense-aware attack, w.r.t. the detector with few fixed features. Theoretical and numerical analyses demonstrate the uniqueness and usefulness of our detector, exhibiting competitive scores on several deep models and image sets against a variety of adversarial attacks.



## **37. A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems**

cs.RO

**SubmitDate**: 2024-08-07    [abs](http://arxiv.org/abs/2408.03515v1) [paper-pdf](http://arxiv.org/pdf/2408.03515v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Braunl, Jin B. Hong

**Abstract**: The integration of Large Language Models (LLMs) like GPT-4o into robotic systems represents a significant advancement in embodied artificial intelligence. These models can process multi-modal prompts, enabling them to generate more context-aware responses. However, this integration is not without challenges. One of the primary concerns is the potential security risks associated with using LLMs in robotic navigation tasks. These tasks require precise and reliable responses to ensure safe and effective operation. Multi-modal prompts, while enhancing the robot's understanding, also introduce complexities that can be exploited maliciously. For instance, adversarial inputs designed to mislead the model can lead to incorrect or dangerous navigational decisions. This study investigates the impact of prompt injections on mobile robot performance in LLM-integrated systems and explores secure prompt strategies to mitigate these risks. Our findings demonstrate a substantial overall improvement of approximately 30.8% in both attack detection and system performance with the implementation of robust defence mechanisms, highlighting their critical role in enhancing security and reliability in mission-oriented tasks.



## **38. Simple Perturbations Subvert Ethereum Phishing Transactions Detection: An Empirical Analysis**

cs.CR

12 pages, 1 figure, 5 tables, accepted for presentation at WISA 2024

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.03441v1) [paper-pdf](http://arxiv.org/pdf/2408.03441v1)

**Authors**: Ahod Alghureid, David Mohaisen

**Abstract**: This paper explores the vulnerability of machine learning models, specifically Random Forest, Decision Tree, and K-Nearest Neighbors, to very simple single-feature adversarial attacks in the context of Ethereum fraudulent transaction detection. Through comprehensive experimentation, we investigate the impact of various adversarial attack strategies on model performance metrics, such as accuracy, precision, recall, and F1-score. Our findings, highlighting how prone those techniques are to simple attacks, are alarming, and the inconsistency in the attacks' effect on different algorithms promises ways for attack mitigation. We examine the effectiveness of different mitigation strategies, including adversarial training and enhanced feature selection, in enhancing model robustness.



## **39. Attacks and Defenses for Generative Diffusion Models: A Comprehensive Survey**

cs.CR

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.03400v1) [paper-pdf](http://arxiv.org/pdf/2408.03400v1)

**Authors**: Vu Tuan Truong, Luan Ba Dang, Long Bao Le

**Abstract**: Diffusion models (DMs) have achieved state-of-the-art performance on various generative tasks such as image synthesis, text-to-image, and text-guided image-to-image generation. However, the more powerful the DMs, the more harmful they potentially are. Recent studies have shown that DMs are prone to a wide range of attacks, including adversarial attacks, membership inference, backdoor injection, and various multi-modal threats. Since numerous pre-trained DMs are published widely on the Internet, potential threats from these attacks are especially detrimental to the society, making DM-related security a worth investigating topic. Therefore, in this paper, we conduct a comprehensive survey on the security aspect of DMs, focusing on various attack and defense methods for DMs. First, we present crucial knowledge of DMs with five main types of DMs, including denoising diffusion probabilistic models, denoising diffusion implicit models, noise conditioned score networks, stochastic differential equations, and multi-modal conditional DMs. We further survey a variety of recent studies investigating different types of attacks that exploit the vulnerabilities of DMs. Then, we thoroughly review potential countermeasures to mitigate each of the presented threats. Finally, we discuss open challenges of DM-related security and envision certain research directions for this topic.



## **40. Self-Evaluation as a Defense Against Adversarial Attacks on LLMs**

cs.LG

8 pages, 7 figures

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2407.03234v3) [paper-pdf](http://arxiv.org/pdf/2407.03234v3)

**Authors**: Hannah Brown, Leon Lin, Kenji Kawaguchi, Michael Shieh

**Abstract**: We introduce a defense against adversarial attacks on LLMs utilizing self-evaluation. Our method requires no model fine-tuning, instead using pre-trained models to evaluate the inputs and outputs of a generator model, significantly reducing the cost of implementation in comparison to other, finetuning-based methods. Our method can significantly reduce the attack success rate of attacks on both open and closed-source LLMs, beyond the reductions demonstrated by Llama-Guard2 and commonly used content moderation APIs. We present an analysis of the effectiveness of our method, including attempts to attack the evaluator in various settings, demonstrating that it is also more resilient to attacks than existing methods. Code and data will be made available at https://github.com/Linlt-leon/self-eval.



## **41. Sample-agnostic Adversarial Perturbation for Vision-Language Pre-training Models**

cs.CV

13 pages, 8 figures, published in ACMMM2024

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.02980v1) [paper-pdf](http://arxiv.org/pdf/2408.02980v1)

**Authors**: Haonan Zheng, Wen Jiang, Xinyang Deng, Wenrui Li

**Abstract**: Recent studies on AI security have highlighted the vulnerability of Vision-Language Pre-training (VLP) models to subtle yet intentionally designed perturbations in images and texts. Investigating multimodal systems' robustness via adversarial attacks is crucial in this field. Most multimodal attacks are sample-specific, generating a unique perturbation for each sample to construct adversarial samples. To the best of our knowledge, it is the first work through multimodal decision boundaries to explore the creation of a universal, sample-agnostic perturbation that applies to any image. Initially, we explore strategies to move sample points beyond the decision boundaries of linear classifiers, refining the algorithm to ensure successful attacks under the top $k$ accuracy metric. Based on this foundation, in visual-language tasks, we treat visual and textual modalities as reciprocal sample points and decision hyperplanes, guiding image embeddings to traverse text-constructed decision boundaries, and vice versa. This iterative process consistently refines a universal perturbation, ultimately identifying a singular direction within the input space which is exploitable to impair the retrieval performance of VLP models. The proposed algorithms support the creation of global perturbations or adversarial patches. Comprehensive experiments validate the effectiveness of our method, showcasing its data, task, and model transferability across various VLP models and datasets. Code: https://github.com/LibertazZ/MUAP



## **42. Sharpness-Aware Cross-Domain Recommendation to Cold-Start Users**

cs.IR

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.01931v2) [paper-pdf](http://arxiv.org/pdf/2408.01931v2)

**Authors**: Guohang Zeng, Qian Zhang, Guangquan Zhang, Jie Lu

**Abstract**: Cross-Domain Recommendation (CDR) is a promising paradigm inspired by transfer learning to solve the cold-start problem in recommender systems. Existing state-of-the-art CDR methods train an explicit mapping function to transfer the cold-start users from a data-rich source domain to a target domain. However, a limitation of these methods is that the mapping function is trained on overlapping users across domains, while only a small number of overlapping users are available for training. By visualizing the loss landscape of the existing CDR model, we find that training on a small number of overlapping users causes the model to converge to sharp minima, leading to poor generalization. Based on this observation, we leverage loss-geometry-based machine learning approach and propose a novel CDR method called Sharpness-Aware CDR (SCDR). Our proposed method simultaneously optimizes recommendation loss and loss sharpness, leading to better generalization with theoretical guarantees. Empirical studies on real-world datasets demonstrate that SCDR significantly outperforms the other CDR models for cold-start recommendation tasks, while concurrently enhancing the model's robustness to adversarial attacks.



## **43. Adversarial Robustness of Open-source Text Classification Models and Fine-Tuning Chains**

cs.SE

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.02963v1) [paper-pdf](http://arxiv.org/pdf/2408.02963v1)

**Authors**: Hao Qin, Mingyang Li, Junjie Wang, Qing Wang

**Abstract**: Context:With the advancement of artificial intelligence (AI) technology and applications, numerous AI models have been developed, leading to the emergence of open-source model hosting platforms like Hugging Face (HF). Thanks to these platforms, individuals can directly download and use models, as well as fine-tune them to construct more domain-specific models. However, just like traditional software supply chains face security risks, AI models and fine-tuning chains also encounter new security risks, such as adversarial attacks. Therefore, the adversarial robustness of these models has garnered attention, potentially influencing people's choices regarding open-source models. Objective:This paper aims to explore the adversarial robustness of open-source AI models and their chains formed by the upstream-downstream relationships via fine-tuning to provide insights into the potential adversarial risks. Method:We collect text classification models on HF and construct the fine-tuning chains.Then, we conduct an empirical analysis of model reuse and associated robustness risks under existing adversarial attacks from two aspects, i.e., models and their fine-tuning chains. Results:Despite the models' widespread downloading and reuse, they are generally susceptible to adversarial attack risks, with an average of 52.70% attack success rate. Moreover, fine-tuning typically exacerbates this risk, resulting in an average 12.60% increase in attack success rates. We also delve into the influence of factors such as attack techniques, datasets, and model architectures on the success rate, as well as the transitivity along the model chains.



## **44. Adv3D: Generating 3D Adversarial Examples for 3D Object Detection in Driving Scenarios with NeRF**

cs.CV

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2309.01351v2) [paper-pdf](http://arxiv.org/pdf/2309.01351v2)

**Authors**: Leheng Li, Qing Lian, Ying-Cong Chen

**Abstract**: Deep neural networks (DNNs) have been proven extremely susceptible to adversarial examples, which raises special safety-critical concerns for DNN-based autonomous driving stacks (i.e., 3D object detection). Although there are extensive works on image-level attacks, most are restricted to 2D pixel spaces, and such attacks are not always physically realistic in our 3D world. Here we present Adv3D, the first exploration of modeling adversarial examples as Neural Radiance Fields (NeRFs). Advances in NeRF provide photorealistic appearances and 3D accurate generation, yielding a more realistic and realizable adversarial example. We train our adversarial NeRF by minimizing the surrounding objects' confidence predicted by 3D detectors on the training set. Then we evaluate Adv3D on the unseen validation set and show that it can cause a large performance reduction when rendering NeRF in any sampled pose. To generate physically realizable adversarial examples, we propose primitive-aware sampling and semantic-guided regularization that enable 3D patch attacks with camouflage adversarial texture. Experimental results demonstrate that the trained adversarial NeRF generalizes well to different poses, scenes, and 3D detectors. Finally, we provide a defense method to our attacks that involves adversarial training through data augmentation. Project page: https://len-li.github.io/adv3d-web



## **45. A Survey and Evaluation of Adversarial Attacks for Object Detection**

cs.CV

14 pages

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.01934v2) [paper-pdf](http://arxiv.org/pdf/2408.01934v2)

**Authors**: Khoi Nguyen Tiet Nguyen, Wenyu Zhang, Kangkang Lu, Yuhuan Wu, Xingjian Zheng, Hui Li Tan, Liangli Zhen

**Abstract**: Deep learning models excel in various computer vision tasks but are susceptible to adversarial examples-subtle perturbations in input data that lead to incorrect predictions. This vulnerability poses significant risks in safety-critical applications such as autonomous vehicles, security surveillance, and aircraft health monitoring. While numerous surveys focus on adversarial attacks in image classification, the literature on such attacks in object detection is limited. This paper offers a comprehensive taxonomy of adversarial attacks specific to object detection, reviews existing adversarial robustness evaluation metrics, and systematically assesses open-source attack methods and model robustness. Key observations are provided to enhance the understanding of attack effectiveness and corresponding countermeasures. Additionally, we identify crucial research challenges to guide future efforts in securing automated object detection systems.



## **46. Compromising Embodied Agents with Contextual Backdoor Attacks**

cs.AI

**SubmitDate**: 2024-08-06    [abs](http://arxiv.org/abs/2408.02882v1) [paper-pdf](http://arxiv.org/pdf/2408.02882v1)

**Authors**: Aishan Liu, Yuguang Zhou, Xianglong Liu, Tianyuan Zhang, Siyuan Liang, Jiakai Wang, Yanjun Pu, Tianlin Li, Junqi Zhang, Wenbo Zhou, Qing Guo, Dacheng Tao

**Abstract**: Large language models (LLMs) have transformed the development of embodied intelligence. By providing a few contextual demonstrations, developers can utilize the extensive internal knowledge of LLMs to effortlessly translate complex tasks described in abstract language into sequences of code snippets, which will serve as the execution logic for embodied agents. However, this paper uncovers a significant backdoor security threat within this process and introduces a novel method called \method{}. By poisoning just a few contextual demonstrations, attackers can covertly compromise the contextual environment of a black-box LLM, prompting it to generate programs with context-dependent defects. These programs appear logically sound but contain defects that can activate and induce unintended behaviors when the operational agent encounters specific triggers in its interactive environment. To compromise the LLM's contextual environment, we employ adversarial in-context generation to optimize poisoned demonstrations, where an LLM judge evaluates these poisoned prompts, reporting to an additional LLM that iteratively optimizes the demonstration in a two-player adversarial game using chain-of-thought reasoning. To enable context-dependent behaviors in downstream agents, we implement a dual-modality activation strategy that controls both the generation and execution of program defects through textual and visual triggers. We expand the scope of our attack by developing five program defect modes that compromise key aspects of confidentiality, integrity, and availability in embodied agents. To validate the effectiveness of our approach, we conducted extensive experiments across various tasks, including robot planning, robot manipulation, and compositional visual reasoning. Additionally, we demonstrate the potential impact of our approach by successfully attacking real-world autonomous driving systems.



## **47. Pre-trained Encoder Inference: Revealing Upstream Encoders In Downstream Machine Learning Services**

cs.LG

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02814v1) [paper-pdf](http://arxiv.org/pdf/2408.02814v1)

**Authors**: Shaopeng Fu, Xuexue Sun, Ke Qing, Tianhang Zheng, Di Wang

**Abstract**: Though pre-trained encoders can be easily accessed online to build downstream machine learning (ML) services quickly, various attacks have been designed to compromise the security and privacy of these encoders. While most attacks target encoders on the upstream side, it remains unknown how an encoder could be threatened when deployed in a downstream ML service. This paper unveils a new vulnerability: the Pre-trained Encoder Inference (PEI) attack, which posts privacy threats toward encoders hidden behind downstream ML services. By only providing API accesses to a targeted downstream service and a set of candidate encoders, the PEI attack can infer which encoder is secretly used by the targeted service based on candidate ones. We evaluate the attack performance of PEI against real-world encoders on three downstream tasks: image classification, text classification, and text-to-image generation. Experiments show that the PEI attack succeeds in revealing the hidden encoder in most cases and seldom makes mistakes even when the hidden encoder is not in the candidate set. We also conducted a case study on one of the most recent vision-language models, LLaVA, to illustrate that the PEI attack is useful in assisting other ML attacks such as adversarial attacks. The code is available at https://github.com/fshp971/encoder-inference.



## **48. SEAS: Self-Evolving Adversarial Safety Optimization for Large Language Models**

cs.CL

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2408.02632v1) [paper-pdf](http://arxiv.org/pdf/2408.02632v1)

**Authors**: Muxi Diao, Rumei Li, Shiyang Liu, Guogang Liao, Jingang Wang, Xunliang Cai, Weiran Xu

**Abstract**: As large language models (LLMs) continue to advance in capability and influence, ensuring their security and preventing harmful outputs has become crucial. A promising approach to address these concerns involves training models to automatically generate adversarial prompts for red teaming. However, the evolving subtlety of vulnerabilities in LLMs challenges the effectiveness of current adversarial methods, which struggle to specifically target and explore the weaknesses of these models. To tackle these challenges, we introduce the $\mathbf{S}\text{elf-}\mathbf{E}\text{volving }\mathbf{A}\text{dversarial }\mathbf{S}\text{afety }\mathbf{(SEAS)}$ optimization framework, which enhances security by leveraging data generated by the model itself. SEAS operates through three iterative stages: Initialization, Attack, and Adversarial Optimization, refining both the Red Team and Target models to improve robustness and safety. This framework reduces reliance on manual testing and significantly enhances the security capabilities of LLMs. Our contributions include a novel adversarial framework, a comprehensive safety dataset, and after three iterations, the Target model achieves a security level comparable to GPT-4, while the Red Team model shows a marked increase in attack success rate (ASR) against advanced models.



## **49. SSAP: A Shape-Sensitive Adversarial Patch for Comprehensive Disruption of Monocular Depth Estimation in Autonomous Navigation Applications**

cs.CV

arXiv admin note: text overlap with arXiv:2303.01351

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2403.11515v2) [paper-pdf](http://arxiv.org/pdf/2403.11515v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Ihsen Alouani, Bassem Ouni, Muhammad Shafique

**Abstract**: Monocular depth estimation (MDE) has advanced significantly, primarily through the integration of convolutional neural networks (CNNs) and more recently, Transformers. However, concerns about their susceptibility to adversarial attacks have emerged, especially in safety-critical domains like autonomous driving and robotic navigation. Existing approaches for assessing CNN-based depth prediction methods have fallen short in inducing comprehensive disruptions to the vision system, often limited to specific local areas. In this paper, we introduce SSAP (Shape-Sensitive Adversarial Patch), a novel approach designed to comprehensively disrupt monocular depth estimation (MDE) in autonomous navigation applications. Our patch is crafted to selectively undermine MDE in two distinct ways: by distorting estimated distances or by creating the illusion of an object disappearing from the system's perspective. Notably, our patch is shape-sensitive, meaning it considers the specific shape and scale of the target object, thereby extending its influence beyond immediate proximity. Furthermore, our patch is trained to effectively address different scales and distances from the camera. Experimental results demonstrate that our approach induces a mean depth estimation error surpassing 0.5, impacting up to 99% of the targeted region for CNN-based MDE models. Additionally, we investigate the vulnerability of Transformer-based MDE models to patch-based attacks, revealing that SSAP yields a significant error of 0.59 and exerts substantial influence over 99% of the target region on these models.



## **50. APARATE: Adaptive Adversarial Patch for CNN-based Monocular Depth Estimation for Autonomous Navigation**

cs.CV

**SubmitDate**: 2024-08-05    [abs](http://arxiv.org/abs/2303.01351v3) [paper-pdf](http://arxiv.org/pdf/2303.01351v3)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Ihsen Alouani, Muhammad Shafique

**Abstract**: In recent times, monocular depth estimation (MDE) has experienced significant advancements in performance, largely attributed to the integration of innovative architectures, i.e., convolutional neural networks (CNNs) and Transformers. Nevertheless, the susceptibility of these models to adversarial attacks has emerged as a noteworthy concern, especially in domains where safety and security are paramount. This concern holds particular weight for MDE due to its critical role in applications like autonomous driving and robotic navigation, where accurate scene understanding is pivotal. To assess the vulnerability of CNN-based depth prediction methods, recent work tries to design adversarial patches against MDE. However, the existing approaches fall short of inducing a comprehensive and substantially disruptive impact on the vision system. Instead, their influence is partial and confined to specific local areas. These methods lead to erroneous depth predictions only within the overlapping region with the input image, without considering the characteristics of the target object, such as its size, shape, and position. In this paper, we introduce a novel adversarial patch named APARATE. This patch possesses the ability to selectively undermine MDE in two distinct ways: by distorting the estimated distances or by creating the illusion of an object disappearing from the perspective of the autonomous system. Notably, APARATE is designed to be sensitive to the shape and scale of the target object, and its influence extends beyond immediate proximity. APARATE, results in a mean depth estimation error surpassing $0.5$, significantly impacting as much as $99\%$ of the targeted region when applied to CNN-based MDE models. Furthermore, it yields a significant error of $0.34$ and exerts substantial influence over $94\%$ of the target region in the context of Transformer-based MDE.



