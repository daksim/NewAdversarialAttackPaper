# Latest Adversarial Attack Papers
**update at 2024-01-27 11:18:49**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Defending Against Physical Adversarial Patch Attacks on Infrared Human Detection**

cs.CV

Lukas Strack and Futa Waseda contributed equally. 6 pages,  Under-review

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2309.15519v2) [paper-pdf](http://arxiv.org/pdf/2309.15519v2)

**Authors**: Lukas Strack, Futa Waseda, Huy H. Nguyen, Yinqiang Zheng, Isao Echizen

**Abstract**: Infrared detection is an emerging technique for safety-critical tasks owing to its remarkable anti-interference capability. However, recent studies have revealed that it is vulnerable to physically-realizable adversarial patches, posing risks in its real-world applications. To address this problem, we are the first to investigate defense strategies against adversarial patch attacks on infrared detection, especially human detection. We have devised a straightforward defense strategy, patch-based occlusion-aware detection (POD), which efficiently augments training samples with random patches and subsequently detects them. POD not only robustly detects people but also identifies adversarial patch locations. Surprisingly, while being extremely computationally efficient, POD easily generalizes to state-of-the-art adversarial patch attacks that are unseen during training. Furthermore, POD improves detection precision even in a clean (i.e., no-attack) situation due to the data augmentation effect. Evaluation demonstrated that POD is robust to adversarial patches of various shapes and sizes. The effectiveness of our baseline approach is shown to be a viable defense mechanism for real-world infrared human detection systems, paving the way for exploring future research directions.



## **2. TrojFST: Embedding Trojans in Few-shot Prompt Tuning**

cs.LG

9 pages

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2312.10467v2) [paper-pdf](http://arxiv.org/pdf/2312.10467v2)

**Authors**: Mengxin Zheng, Jiaqi Xue, Xun Chen, YanShan Wang, Qian Lou, Lei Jiang

**Abstract**: Prompt-tuning has emerged as a highly effective approach for adapting a pre-trained language model (PLM) to handle new natural language processing tasks with limited input samples. However, the success of prompt-tuning has led to adversaries attempting backdoor attacks against this technique. Previous prompt-based backdoor attacks faced challenges when implemented through few-shot prompt-tuning, requiring either full-model fine-tuning or a large training dataset. We observe the difficulty in constructing a prompt-based backdoor using few-shot prompt-tuning, which involves freezing the PLM and tuning a soft prompt with a restricted set of input samples. This approach introduces an imbalanced poisoned dataset, making it susceptible to overfitting and lacking attention awareness. To address these challenges, we introduce TrojFST for backdoor attacks within the framework of few-shot prompt-tuning. TrojFST comprises three modules: balanced poison learning, selective token poisoning, and trojan-trigger attention. In comparison to previous prompt-based backdoor attacks, TrojFST demonstrates significant improvements, enhancing ASR $> 9\%$ and CDA by $> 4\%$ across various PLMs and a diverse set of downstream tasks.



## **3. The Surprising Harmfulness of Benign Overfitting for Adversarial Robustness**

cs.LG

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2401.12236v2) [paper-pdf](http://arxiv.org/pdf/2401.12236v2)

**Authors**: Yifan Hao, Tong Zhang

**Abstract**: Recent empirical and theoretical studies have established the generalization capabilities of large machine learning models that are trained to (approximately or exactly) fit noisy data. In this work, we prove a surprising result that even if the ground truth itself is robust to adversarial examples, and the benignly overfitted model is benign in terms of the ``standard'' out-of-sample risk objective, this benign overfitting process can be harmful when out-of-sample data are subject to adversarial manipulation. More specifically, our main results contain two parts: (i) the min-norm estimator in overparameterized linear model always leads to adversarial vulnerability in the ``benign overfitting'' setting; (ii) we verify an asymptotic trade-off result between the standard risk and the ``adversarial'' risk of every ridge regression estimator, implying that under suitable conditions these two items cannot both be small at the same time by any single choice of the ridge regularization parameter. Furthermore, under the lazy training regime, we demonstrate parallel results on two-layer neural tangent kernel (NTK) model, which align with empirical observations in deep neural networks. Our finding provides theoretical insights into the puzzling phenomenon observed in practice, where the true target function (e.g., human) is robust against adverasrial attack, while beginly overfitted neural networks lead to models that are not robust.



## **4. Friendly Attacks to Improve Channel Coding Reliability**

cs.IT

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2401.14184v1) [paper-pdf](http://arxiv.org/pdf/2401.14184v1)

**Authors**: Anastasiia Kurmukova, Deniz Gunduz

**Abstract**: This paper introduces a novel approach called "friendly attack" aimed at enhancing the performance of error correction channel codes. Inspired by the concept of adversarial attacks, our method leverages the idea of introducing slight perturbations to the neural network input, resulting in a substantial impact on the network's performance. By introducing small perturbations to fixed-point modulated codewords before transmission, we effectively improve the decoder's performance without violating the input power constraint. The perturbation design is accomplished by a modified iterative fast gradient method. This study investigates various decoder architectures suitable for computing gradients to obtain the desired perturbations. Specifically, we consider belief propagation (BP) for LDPC codes; the error correcting code transformer, BP and neural BP (NBP) for polar codes, and neural BCJR for convolutional codes. We demonstrate that the proposed friendly attack method can improve the reliability across different channels, modulations, codes, and decoders. This method allows us to increase the reliability of communication with a legacy receiver by simply modifying the transmitted codeword appropriately.



## **5. EvadeDroid: A Practical Evasion Attack on Machine Learning for Black-box Android Malware Detection**

cs.LG

The paper was accepted by Elsevier Computers & Security on 20  December 2023

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2110.03301v4) [paper-pdf](http://arxiv.org/pdf/2110.03301v4)

**Authors**: Hamid Bostani, Veelasha Moonsamy

**Abstract**: Over the last decade, researchers have extensively explored the vulnerabilities of Android malware detectors to adversarial examples through the development of evasion attacks; however, the practicality of these attacks in real-world scenarios remains arguable. The majority of studies have assumed attackers know the details of the target classifiers used for malware detection, while in reality, malicious actors have limited access to the target classifiers. This paper introduces EvadeDroid, a problem-space adversarial attack designed to effectively evade black-box Android malware detectors in real-world scenarios. EvadeDroid constructs a collection of problem-space transformations derived from benign donors that share opcode-level similarity with malware apps by leveraging an n-gram-based approach. These transformations are then used to morph malware instances into benign ones via an iterative and incremental manipulation strategy. The proposed manipulation technique is a query-efficient optimization algorithm that can find and inject optimal sequences of transformations into malware apps. Our empirical evaluations, carried out on 1K malware apps, demonstrate the effectiveness of our approach in generating real-world adversarial examples in both soft- and hard-label settings. Our findings reveal that EvadeDroid can effectively deceive diverse malware detectors that utilize different features with various feature types. Specifically, EvadeDroid achieves evasion rates of 80%-95% against DREBIN, Sec-SVM, ADE-MA, MaMaDroid, and Opcode-SVM with only 1-9 queries. Furthermore, we show that the proposed problem-space adversarial attack is able to preserve its stealthiness against five popular commercial antiviruses with an average of 79% evasion rate, thus demonstrating its feasibility in the real world.



## **6. Username Squatting on Online Social Networks: A Study on X**

cs.CR

Accepted at the 19th ACM ASIA Conference on Computer and  Communications Security (ACM ASIACCS 2024)

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2401.09209v2) [paper-pdf](http://arxiv.org/pdf/2401.09209v2)

**Authors**: Anastasios Lepipas, Anastasia Borovykh, Soteris Demetriou

**Abstract**: Adversaries have been targeting unique identifiers to launch typo-squatting, mobile app squatting and even voice squatting attacks. Anecdotal evidence suggest that online social networks (OSNs) are also plagued with accounts that use similar usernames. This can be confusing to users but can also be exploited by adversaries. However, to date no study characterizes this problem on OSNs. In this work, we define the username squatting problem and design the first multi-faceted measurement study to characterize it on X. We develop a username generation tool (UsernameCrazy) to help us analyze hundreds of thousands of username variants derived from celebrity accounts. Our study reveals that thousands of squatted usernames have been suspended by X, while tens of thousands that still exist on the network are likely bots. Out of these, a large number share similar profile pictures and profile names to the original account signalling impersonation attempts. We found that squatted accounts are being mentioned by mistake in tweets hundreds of thousands of times and are even being prioritized in searches by the network's search recommendation algorithm exacerbating the negative impact squatted accounts can have in OSNs. We use our insights and take the first step to address this issue by designing a framework (SQUAD) that combines UsernameCrazy with a new classifier to efficiently detect suspicious squatted accounts. Our evaluation of SQUAD's prototype implementation shows that it can achieve 94% F1-score when trained on a small dataset.



## **7. Sparse and Transferable Universal Singular Vectors Attack**

cs.LG

**SubmitDate**: 2024-01-25    [abs](http://arxiv.org/abs/2401.14031v1) [paper-pdf](http://arxiv.org/pdf/2401.14031v1)

**Authors**: Kseniia Kuvshinova, Olga Tsymboi, Ivan Oseledets

**Abstract**: The research in the field of adversarial attacks and models' vulnerability is one of the fundamental directions in modern machine learning. Recent studies reveal the vulnerability phenomenon, and understanding the mechanisms behind this is essential for improving neural network characteristics and interpretability. In this paper, we propose a novel sparse universal white-box adversarial attack. Our approach is based on truncated power iteration providing sparsity to $(p,q)$-singular vectors of the hidden layers of Jacobian matrices. Using the ImageNet benchmark validation subset, we analyze the proposed method in various settings, achieving results comparable to dense baselines with more than a 50% fooling rate while damaging only 5% of pixels and utilizing 256 samples for perturbation fitting. We also show that our algorithm admits higher attack magnitude without affecting the human ability to solve the task. Furthermore, we investigate that the constructed perturbations are highly transferable among different models without significantly decreasing the fooling rate. Our findings demonstrate the vulnerability of state-of-the-art models to sparse attacks and highlight the importance of developing robust machine learning systems.



## **8. Exploring Adversarial Threat Models in Cyber Physical Battery Systems**

eess.SY

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2401.13801v1) [paper-pdf](http://arxiv.org/pdf/2401.13801v1)

**Authors**: Shanthan Kumar Padisala, Shashank Dhananjay Vyas, Satadru Dey

**Abstract**: Technological advancements like the Internet of Things (IoT) have facilitated data exchange across various platforms. This data exchange across various platforms has transformed the traditional battery system into a cyber physical system. Such connectivity makes modern cyber physical battery systems vulnerable to cyber threats where a cyber attacker can manipulate sensing and actuation signals to bring the battery system into an unsafe operating condition. Hence, it is essential to build resilience in modern cyber physical battery systems (CPBS) under cyber attacks. The first step of building such resilience is to analyze potential adversarial behavior, that is, how the adversaries can inject attacks into the battery systems. However, it has been found that in this under-explored area of battery cyber physical security, such an adversarial threat model has not been studied in a systematic manner. In this study, we address this gap and explore adversarial attack generation policies based on optimal control framework. The framework is developed by performing theoretical analysis, which is subsequently supported by evaluation with experimental data generated from a commercial battery cell.



## **9. A Systematic Approach to Robustness Modelling for Deep Convolutional Neural Networks**

cs.LG

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2401.13751v1) [paper-pdf](http://arxiv.org/pdf/2401.13751v1)

**Authors**: Charles Meyers, Mohammad Reza Saleh Sedghpour, Tommy Löfstedt, Erik Elmroth

**Abstract**: Convolutional neural networks have shown to be widely applicable to a large number of fields when large amounts of labelled data are available. The recent trend has been to use models with increasingly larger sets of tunable parameters to increase model accuracy, reduce model loss, or create more adversarially robust models -- goals that are often at odds with one another. In particular, recent theoretical work raises questions about the ability for even larger models to generalize to data outside of the controlled train and test sets. As such, we examine the role of the number of hidden layers in the ResNet model, demonstrated on the MNIST, CIFAR10, CIFAR100 datasets. We test a variety of parameters including the size of the model, the floating point precision, and the noise level of both the training data and the model output. To encapsulate the model's predictive power and computational cost, we provide a method that uses induced failures to model the probability of failure as a function of time and relate that to a novel metric that allows us to quickly determine whether or not the cost of training a model outweighs the cost of attacking it. Using this approach, we are able to approximate the expected failure rate using a small number of specially crafted samples rather than increasingly larger benchmark datasets. We demonstrate the efficacy of this technique on both the MNIST and CIFAR10 datasets using 8-, 16-, 32-, and 64-bit floating-point numbers, various data pre-processing techniques, and several attacks on five configurations of the ResNet model. Then, using empirical measurements, we examine the various trade-offs between cost, robustness, latency, and reliability to find that larger models do not significantly aid in adversarial robustness despite costing significantly more to train.



## **10. TrojanPuzzle: Covertly Poisoning Code-Suggestion Models**

cs.CR

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2301.02344v2) [paper-pdf](http://arxiv.org/pdf/2301.02344v2)

**Authors**: Hojjat Aghakhani, Wei Dai, Andre Manoel, Xavier Fernandes, Anant Kharkar, Christopher Kruegel, Giovanni Vigna, David Evans, Ben Zorn, Robert Sim

**Abstract**: With tools like GitHub Copilot, automatic code suggestion is no longer a dream in software engineering. These tools, based on large language models, are typically trained on massive corpora of code mined from unvetted public sources. As a result, these models are susceptible to data poisoning attacks where an adversary manipulates the model's training by injecting malicious data. Poisoning attacks could be designed to influence the model's suggestions at run time for chosen contexts, such as inducing the model into suggesting insecure code payloads. To achieve this, prior attacks explicitly inject the insecure code payload into the training data, making the poison data detectable by static analysis tools that can remove such malicious data from the training set. In this work, we demonstrate two novel attacks, COVERT and TROJANPUZZLE, that can bypass static analysis by planting malicious poison data in out-of-context regions such as docstrings. Our most novel attack, TROJANPUZZLE, goes one step further in generating less suspicious poison data by never explicitly including certain (suspicious) parts of the payload in the poison data, while still inducing a model that suggests the entire payload when completing code (i.e., outside docstrings). This makes TROJANPUZZLE robust against signature-based dataset-cleansing methods that can filter out suspicious sequences from the training data. Our evaluation against models of two sizes demonstrates that both COVERT and TROJANPUZZLE have significant implications for practitioners when selecting code used to train or tune code-suggestion models.



## **11. Synthesizing Physical Backdoor Datasets: An Automated Framework Leveraging Deep Generative Models**

cs.CR

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2312.03419v2) [paper-pdf](http://arxiv.org/pdf/2312.03419v2)

**Authors**: Sze Jue Yang, Chinh D. La, Quang H. Nguyen, Eugene Bagdasaryan, Kok-Seng Wong, Anh Tuan Tran, Chee Seng Chan, Khoa D. Doan

**Abstract**: Backdoor attacks, representing an emerging threat to the integrity of deep neural networks, have garnered significant attention due to their ability to compromise deep learning systems clandestinely. While numerous backdoor attacks occur within the digital realm, their practical implementation in real-world prediction systems remains limited and vulnerable to disturbances in the physical world. Consequently, this limitation has given rise to the development of physical backdoor attacks, where trigger objects manifest as physical entities within the real world. However, creating the requisite dataset to train or evaluate a physical backdoor model is a daunting task, limiting the backdoor researchers and practitioners from studying such physical attack scenarios. This paper unleashes a recipe that empowers backdoor researchers to effortlessly create a malicious, physical backdoor dataset based on advances in generative modeling. Particularly, this recipe involves 3 automatic modules: suggesting the suitable physical triggers, generating the poisoned candidate samples (either by synthesizing new samples or editing existing clean samples), and finally refining for the most plausible ones. As such, it effectively mitigates the perceived complexity associated with creating a physical backdoor dataset, transforming it from a daunting task into an attainable objective. Extensive experiment results show that datasets created by our "recipe" enable adversaries to achieve an impressive attack success rate on real physical world data and exhibit similar properties compared to previous physical backdoor attack studies. This paper offers researchers a valuable toolkit for studies of physical backdoors, all within the confines of their laboratories.



## **12. Adversarial Detection by Approximation of Ensemble Boundary**

cs.LG

17 pages, 5 figures, 5 tables

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2211.10227v4) [paper-pdf](http://arxiv.org/pdf/2211.10227v4)

**Authors**: T. Windeatt

**Abstract**: A new method of detecting adversarial attacks is proposed for an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The ensemble is combined using Walsh coefficients which are capable of approximating Boolean functions and thereby controlling the complexity of the ensemble decision boundary. The hypothesis in this paper is that decision boundaries with high curvature allow adversarial perturbations to be found, but change the curvature of the decision boundary, which is then approximated in a different way by Walsh coefficients compared to the clean images. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it is shown experimentally that transferability of attack may be used for detection. Furthermore, approximating the decision boundary may aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area. Code for approximating Boolean functions using Walsh coefficients: https://doi.org/10.24433/CO.3695905.v1



## **13. Inference Attacks Against Face Recognition Model without Classification Layers**

cs.CV

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2401.13719v1) [paper-pdf](http://arxiv.org/pdf/2401.13719v1)

**Authors**: Yuanqing Huang, Huilong Chen, Yinggui Wang, Lei Wang

**Abstract**: Face recognition (FR) has been applied to nearly every aspect of daily life, but it is always accompanied by the underlying risk of leaking private information. At present, almost all attack models against FR rely heavily on the presence of a classification layer. However, in practice, the FR model can obtain complex features of the input via the model backbone, and then compare it with the target for inference, which does not explicitly involve the outputs of the classification layer adopting logit or other losses. In this work, we advocate a novel inference attack composed of two stages for practical FR models without a classification layer. The first stage is the membership inference attack. Specifically, We analyze the distances between the intermediate features and batch normalization (BN) parameters. The results indicate that this distance is a critical metric for membership inference. We thus design a simple but effective attack model that can determine whether a face image is from the training dataset or not. The second stage is the model inversion attack, where sensitive private data is reconstructed using a pre-trained generative adversarial network (GAN) guided by the attack model in the first stage. To the best of our knowledge, the proposed attack model is the very first in the literature developed for FR models without a classification layer. We illustrate the application of the proposed attack model in the establishment of privacy-preserving FR techniques.



## **14. AdCorDA: Classifier Refinement via Adversarial Correction and Domain Adaptation**

cs.CV

**SubmitDate**: 2024-01-24    [abs](http://arxiv.org/abs/2401.13212v1) [paper-pdf](http://arxiv.org/pdf/2401.13212v1)

**Authors**: Lulan Shen, Ali Edalati, Brett Meyer, Warren Gross, James J. Clark

**Abstract**: This paper describes a simple yet effective technique for refining a pretrained classifier network. The proposed AdCorDA method is based on modification of the training set and making use of the duality between network weights and layer inputs. We call this input space training. The method consists of two stages - adversarial correction followed by domain adaptation. Adversarial correction uses adversarial attacks to correct incorrect training-set classifications. The incorrectly classified samples of the training set are removed and replaced with the adversarially corrected samples to form a new training set, and then, in the second stage, domain adaptation is performed back to the original training set. Extensive experimental validations show significant accuracy boosts of over 5% on the CIFAR-100 dataset. The technique can be straightforwardly applied to refinement of weight-quantized neural networks, where experiments show substantial enhancement in performance over the baseline. The adversarial correction technique also results in enhanced robustness to adversarial attacks.



## **15. How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**

cs.CL

14 pages of the main text, qualitative examples of jailbreaks may be  harmful in nature

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.06373v2) [paper-pdf](http://arxiv.org/pdf/2401.06373v2)

**Authors**: Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi

**Abstract**: Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs



## **16. MAPPING: Debiasing Graph Neural Networks for Fair Node Classification with Limited Sensitive Information Leakage**

cs.LG

Finished May last year. Remember to submit all papers to arXiv early  without compromising the principles of conferences

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.12824v1) [paper-pdf](http://arxiv.org/pdf/2401.12824v1)

**Authors**: Ying Song, Balaji Palanisamy

**Abstract**: Despite remarkable success in diverse web-based applications, Graph Neural Networks(GNNs) inherit and further exacerbate historical discrimination and social stereotypes, which critically hinder their deployments in high-stake domains such as online clinical diagnosis, financial crediting, etc. However, current fairness research that primarily craft on i.i.d data, cannot be trivially replicated to non-i.i.d. graph structures with topological dependence among samples. Existing fair graph learning typically favors pairwise constraints to achieve fairness but fails to cast off dimensional limitations and generalize them into multiple sensitive attributes; besides, most studies focus on in-processing techniques to enforce and calibrate fairness, constructing a model-agnostic debiasing GNN framework at the pre-processing stage to prevent downstream misuses and improve training reliability is still largely under-explored. Furthermore, previous work on GNNs tend to enhance either fairness or privacy individually but few probe into their interplays. In this paper, we propose a novel model-agnostic debiasing framework named MAPPING (\underline{M}asking \underline{A}nd \underline{P}runing and Message-\underline{P}assing train\underline{ING}) for fair node classification, in which we adopt the distance covariance($dCov$)-based fairness constraints to simultaneously reduce feature and topology biases in arbitrary dimensions, and combine them with adversarial debiasing to confine the risks of attribute inference attacks. Experiments on real-world datasets with different GNN variants demonstrate the effectiveness and flexibility of MAPPING. Our results show that MAPPING can achieve better trade-offs between utility and fairness, and mitigate privacy risks of sensitive information leakage.



## **17. The twin peaks of learning neural networks**

cs.LG

36 pages, 30 figures

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.12610v1) [paper-pdf](http://arxiv.org/pdf/2401.12610v1)

**Authors**: Elizaveta Demyanenko, Christoph Feinauer, Enrico M. Malatesta, Luca Saglietti

**Abstract**: Recent works demonstrated the existence of a double-descent phenomenon for the generalization error of neural networks, where highly overparameterized models escape overfitting and achieve good test performance, at odds with the standard bias-variance trade-off described by statistical learning theory. In the present work, we explore a link between this phenomenon and the increase of complexity and sensitivity of the function represented by neural networks. In particular, we study the Boolean mean dimension (BMD), a metric developed in the context of Boolean function analysis. Focusing on a simple teacher-student setting for the random feature model, we derive a theoretical analysis based on the replica method that yields an interpretable expression for the BMD, in the high dimensional regime where the number of data points, the number of features, and the input size grow to infinity. We find that, as the degree of overparameterization of the network is increased, the BMD reaches an evident peak at the interpolation threshold, in correspondence with the generalization error peak, and then slowly approaches a low asymptotic value. The same phenomenology is then traced in numerical experiments with different model classes and training setups. Moreover, we find empirically that adversarially initialized models tend to show higher BMD values, and that models that are more robust to adversarial attacks exhibit a lower BMD.



## **18. ToDA: Target-oriented Diffusion Attacker against Recommendation System**

cs.CR

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.12578v1) [paper-pdf](http://arxiv.org/pdf/2401.12578v1)

**Authors**: Xiaohao Liu, Zhulin Tao, Ting Jiang, He Chang, Yunshan Ma, Xianglin Huang

**Abstract**: Recommendation systems (RS) have become indispensable tools for web services to address information overload, thus enhancing user experiences and bolstering platforms' revenues. However, with their increasing ubiquity, security concerns have also emerged. As the public accessibility of RS, they are susceptible to specific malicious attacks where adversaries can manipulate user profiles, leading to biased recommendations. Recent research often integrates additional modules using generative models to craft these deceptive user profiles, ensuring them are imperceptible while causing the intended harm. Albeit their efficacy, these models face challenges of unstable training and the exploration-exploitation dilemma, which can lead to suboptimal results. In this paper, we pioneer to investigate the potential of diffusion models (DMs), for shilling attacks. Specifically, we propose a novel Target-oriented Diffusion Attack model (ToDA). It incorporates a pre-trained autoencoder that transforms user profiles into a high dimensional space, paired with a Latent Diffusion Attacker (LDA)-the core component of ToDA. LDA introduces noise into the profiles within this latent space, adeptly steering the approximation towards targeted items through cross-attention mechanisms. The global horizon, implemented by a bipartite graph, is involved in LDA and derived from the encoded user profile feature. This makes LDA possible to extend the generation outwards the on-processing user feature itself, and bridges the gap between diffused user features and target item features. Extensive experiments compared to several SOTA baselines demonstrate ToDA's effectiveness. Specific studies exploit the elaborative design of ToDA and underscore the potency of advanced generative models in such contexts.



## **19. Iterative Adversarial Attack on Image-guided Story Ending Generation**

cs.CV

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2305.13208v2) [paper-pdf](http://arxiv.org/pdf/2305.13208v2)

**Authors**: Youze Wang, Wenbo Hu, Richang Hong

**Abstract**: Multimodal learning involves developing models that can integrate information from various sources like images and texts. In this field, multimodal text generation is a crucial aspect that involves processing data from multiple modalities and outputting text. The image-guided story ending generation (IgSEG) is a particularly significant task, targeting on an understanding of complex relationships between text and image data with a complete story text ending. Unfortunately, deep neural networks, which are the backbone of recent IgSEG models, are vulnerable to adversarial samples. Current adversarial attack methods mainly focus on single-modality data and do not analyze adversarial attacks for multimodal text generation tasks that use cross-modal information. To this end, we propose an iterative adversarial attack method (Iterative-attack) that fuses image and text modality attacks, allowing for an attack search for adversarial text and image in an more effective iterative way. Experimental results demonstrate that the proposed method outperforms existing single-modal and non-iterative multimodal attack methods, indicating the potential for improving the adversarial robustness of multimodal text generation models, such as multimodal machine translation, multimodal question answering, etc.



## **20. DAFA: Distance-Aware Fair Adversarial Training**

cs.LG

Accepted to ICLR 2024

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.12532v1) [paper-pdf](http://arxiv.org/pdf/2401.12532v1)

**Authors**: Hyungyu Lee, Saehyung Lee, Hyemi Jang, Junsung Park, Ho Bae, Sungroh Yoon

**Abstract**: The disparity in accuracy between classes in standard training is amplified during adversarial training, a phenomenon termed the robust fairness problem. Existing methodologies aimed to enhance robust fairness by sacrificing the model's performance on easier classes in order to improve its performance on harder ones. However, we observe that under adversarial attacks, the majority of the model's predictions for samples from the worst class are biased towards classes similar to the worst class, rather than towards the easy classes. Through theoretical and empirical analysis, we demonstrate that robust fairness deteriorates as the distance between classes decreases. Motivated by these insights, we introduce the Distance-Aware Fair Adversarial training (DAFA) methodology, which addresses robust fairness by taking into account the similarities between classes. Specifically, our method assigns distinct loss weights and adversarial margins to each class and adjusts them to encourage a trade-off in robustness among similar classes. Experimental results across various datasets demonstrate that our method not only maintains average robust accuracy but also significantly improves the worst robust accuracy, indicating a marked improvement in robust fairness compared to existing methods.



## **21. Explainability-Driven Leaf Disease Classification Using Adversarial Training and Knowledge Distillation**

cs.CV

10 pages, 8 figures, Accepted by ICAART 2024

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.00334v3) [paper-pdf](http://arxiv.org/pdf/2401.00334v3)

**Authors**: Sebastian-Vasile Echim, Iulian-Marius Tăiatu, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: This work focuses on plant leaf disease classification and explores three crucial aspects: adversarial training, model explainability, and model compression. The models' robustness against adversarial attacks is enhanced through adversarial training, ensuring accurate classification even in the presence of threats. Leveraging explainability techniques, we gain insights into the model's decision-making process, improving trust and transparency. Additionally, we explore model compression techniques to optimize computational efficiency while maintaining classification performance. Through our experiments, we determine that on a benchmark dataset, the robustness can be the price of the classification accuracy with performance reductions of 3%-20% for regular tests and gains of 50%-70% for adversarial attack tests. We also demonstrate that a student model can be 15-25 times more computationally efficient for a slight performance reduction, distilling the knowledge of more complex models.



## **22. Fast Adversarial Training against Textual Adversarial Attacks**

cs.CL

4 pages, 4 figures

**SubmitDate**: 2024-01-23    [abs](http://arxiv.org/abs/2401.12461v1) [paper-pdf](http://arxiv.org/pdf/2401.12461v1)

**Authors**: Yichen Yang, Xin Liu, Kun He

**Abstract**: Many adversarial defense methods have been proposed to enhance the adversarial robustness of natural language processing models. However, most of them introduce additional pre-set linguistic knowledge and assume that the synonym candidates used by attackers are accessible, which is an ideal assumption. We delve into adversarial training in the embedding space and propose a Fast Adversarial Training (FAT) method to improve the model robustness in the synonym-unaware scenario from the perspective of single-step perturbation generation and perturbation initialization. Based on the observation that the adversarial perturbations crafted by single-step and multi-step gradient ascent are similar, FAT uses single-step gradient ascent to craft adversarial examples in the embedding space to expedite the training process. Based on the observation that the perturbations generated on the identical training sample in successive epochs are similar, FAT fully utilizes historical information when initializing the perturbation. Extensive experiments demonstrate that FAT significantly boosts the robustness of BERT models in the synonym-unaware scenario, and outperforms the defense baselines under various attacks with character-level and word-level modifications.



## **23. CasTGAN: Cascaded Generative Adversarial Network for Realistic Tabular Data Synthesis**

cs.LG

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2307.00384v2) [paper-pdf](http://arxiv.org/pdf/2307.00384v2)

**Authors**: Abdallah Alshantti, Damiano Varagnolo, Adil Rasheed, Aria Rahmati, Frank Westad

**Abstract**: Generative adversarial networks (GANs) have drawn considerable attention in recent years for their proven capability in generating synthetic data which can be utilised for multiple purposes. While GANs have demonstrated tremendous successes in producing synthetic data samples that replicate the dynamics of the original datasets, the validity of the synthetic data and the underlying privacy concerns represent major challenges which are not sufficiently addressed. In this work, we design a cascaded tabular GAN framework (CasTGAN) for generating realistic tabular data with a specific focus on the validity of the output. In this context, validity refers to the the dependency between features that can be found in the real data, but is typically misrepresented by traditional generative models. Our key idea entails that employing a cascaded architecture in which a dedicated generator samples each feature, the synthetic output becomes more representative of the real data. Our experimental results demonstrate that our model is capable of generating synthetic tabular data that can be used for fitting machine learning models. In addition, our model captures well the constraints and the correlations between the features of the real data, especially the high dimensional datasets. Furthermore, we evaluate the risk of white-box privacy attacks on our model and subsequently show that applying some perturbations to the auxiliary learners in CasTGAN increases the overall robustness of our model against targeted attacks.



## **24. Benchmarking the Robustness of Image Watermarks**

cs.CV

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.08573v2) [paper-pdf](http://arxiv.org/pdf/2401.08573v2)

**Authors**: Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, Furong Huang

**Abstract**: This paper investigates the weaknesses of image watermarking techniques. We present WAVES (Watermark Analysis Via Enhanced Stress-testing), a novel benchmark for assessing watermark robustness, overcoming the limitations of current evaluation methods.WAVES integrates detection and identification tasks, and establishes a standardized evaluation protocol comprised of a diverse range of stress tests. The attacks in WAVES range from traditional image distortions to advanced and novel variations of diffusive, and adversarial attacks. Our evaluation examines two pivotal dimensions: the degree of image quality degradation and the efficacy of watermark detection after attacks. We develop a series of Performance vs. Quality 2D plots, varying over several prominent image similarity metrics, which are then aggregated in a heuristically novel manner to paint an overall picture of watermark robustness and attack potency. Our comprehensive evaluation reveals previously undetected vulnerabilities of several modern watermarking algorithms. We envision WAVES as a toolkit for the future development of robust watermarking systems. The project is available at https://wavesbench.github.io/



## **25. NEUROSEC: FPGA-Based Neuromorphic Audio Security**

cs.CR

Audio processing, FPGA, Hardware Security, Neuromorphic Computing

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.12055v1) [paper-pdf](http://arxiv.org/pdf/2401.12055v1)

**Authors**: Murat Isik, Hiruna Vishwamith, Yusuf Sur, Kayode Inadagbo, I. Can Dikmen

**Abstract**: Neuromorphic systems, inspired by the complexity and functionality of the human brain, have gained interest in academic and industrial attention due to their unparalleled potential across a wide range of applications. While their capabilities herald innovation, it is imperative to underscore that these computational paradigms, analogous to their traditional counterparts, are not impervious to security threats. Although the exploration of neuromorphic methodologies for image and video processing has been rigorously pursued, the realm of neuromorphic audio processing remains in its early stages. Our results highlight the robustness and precision of our FPGA-based neuromorphic system. Specifically, our system showcases a commendable balance between desired signal and background noise, efficient spike rate encoding, and unparalleled resilience against adversarial attacks such as FGSM and PGD. A standout feature of our framework is its detection rate of 94%, which, when compared to other methodologies, underscores its greater capability in identifying and mitigating threats within 5.39 dB, a commendable SNR ratio. Furthermore, neuromorphic computing and hardware security serve many sensor domains in mission-critical and privacy-preserving applications.



## **26. The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images**

cs.CV

ICLR 2024. Code:  https://github.com/mazurowski-lab/intrinsic-properties

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.08865v2) [paper-pdf](http://arxiv.org/pdf/2401.08865v2)

**Authors**: Nicholas Konz, Maciej A. Mazurowski

**Abstract**: This paper investigates discrepancies in how neural networks learn from different imaging domains, which are commonly overlooked when adopting computer vision techniques from the domain of natural images to other specialized domains such as medical images. Recent works have found that the generalization error of a trained network typically increases with the intrinsic dimension ($d_{data}$) of its training set. Yet, the steepness of this relationship varies significantly between medical (radiological) and natural imaging domains, with no existing theoretical explanation. We address this gap in knowledge by establishing and empirically validating a generalization scaling law with respect to $d_{data}$, and propose that the substantial scaling discrepancy between the two considered domains may be at least partially attributed to the higher intrinsic "label sharpness" ($K_F$) of medical imaging datasets, a metric which we propose. Next, we demonstrate an additional benefit of measuring the label sharpness of a training set: it is negatively correlated with the trained model's adversarial robustness, which notably leads to models for medical images having a substantially higher vulnerability to adversarial attack. Finally, we extend our $d_{data}$ formalism to the related metric of learned representation intrinsic dimension ($d_{repr}$), derive a generalization scaling law with respect to $d_{repr}$, and show that $d_{data}$ serves as an upper bound for $d_{repr}$. Our theoretical results are supported by thorough experiments with six models and eleven natural and medical imaging datasets over a range of training set sizes. Our findings offer insights into the influence of intrinsic dataset properties on generalization, representation learning, and robustness in deep neural networks.



## **27. Diagnosis-guided Attack Recovery for Securing Robotic Vehicles from Sensor Deception Attacks**

cs.RO

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2209.04554v5) [paper-pdf](http://arxiv.org/pdf/2209.04554v5)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstract**: Sensors are crucial for perception and autonomous operation in robotic vehicles (RV). Unfortunately, RV sensors can be compromised by physical attacks such as sensor tampering or spoofing. In this paper, we present DeLorean, a unified framework for attack detection, attack diagnosis, and recovering RVs from sensor deception attacks (SDA). DeLorean can recover RVs even from strong SDAs in which the adversary targets multiple heterogeneous sensors simultaneously. We propose a novel attack diagnosis technique that inspects the attack-induced errors under SDAs, and identifies the targeted sensors using causal analysis. DeLorean then uses historic state information to selectively reconstruct physical states for compromised sensors, enabling targeted attack recovery under single or multi-sensor SDAs. We evaluate DeLorean on four real and two simulated RVs under SDAs targeting various sensors, and we find that it successfully recovers RVs from SDAs in 93% of the cases.



## **28. A Training-Free Defense Framework for Robust Learned Image Compression**

eess.IV

10 pages and 14 figures

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11902v1) [paper-pdf](http://arxiv.org/pdf/2401.11902v1)

**Authors**: Myungseo Song, Jinyoung Choi, Bohyung Han

**Abstract**: We study the robustness of learned image compression models against adversarial attacks and present a training-free defense technique based on simple image transform functions. Recent learned image compression models are vulnerable to adversarial attacks that result in poor compression rate, low reconstruction quality, or weird artifacts. To address the limitations, we propose a simple but effective two-way compression algorithm with random input transforms, which is conveniently applicable to existing image compression models. Unlike the na\"ive approaches, our approach preserves the original rate-distortion performance of the models on clean images. Moreover, the proposed algorithm requires no additional training or modification of existing models, making it more practical. We demonstrate the effectiveness of the proposed techniques through extensive experiments under multiple compression models, evaluation metrics, and attack scenarios.



## **29. Adversarial speech for voice privacy protection from Personalized Speech generation**

eess.AS

Accepted by icassp 2024

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11857v1) [paper-pdf](http://arxiv.org/pdf/2401.11857v1)

**Authors**: Shihao Chen, Liping Chen, Jie Zhang, KongAik Lee, Zhenhua Ling, Lirong Dai

**Abstract**: The rapid progress in personalized speech generation technology, including personalized text-to-speech (TTS) and voice conversion (VC), poses a challenge in distinguishing between generated and real speech for human listeners, resulting in an urgent demand in protecting speakers' voices from malicious misuse. In this regard, we propose a speaker protection method based on adversarial attacks. The proposed method perturbs speech signals by minimally altering the original speech while rendering downstream speech generation models unable to accurately generate the voice of the target speaker. For validation, we employ the open-source pre-trained YourTTS model for speech generation and protect the target speaker's speech in the white-box scenario. Automatic speaker verification (ASV) evaluations were carried out on the generated speech as the assessment of the voice protection capability. Our experimental results show that we successfully perturbed the speaker encoder of the YourTTS model using the gradient-based I-FGSM adversarial perturbation method. Furthermore, the adversarial perturbation is effective in preventing the YourTTS model from generating the speech of the target speaker. Audio samples can be found in https://voiceprivacy.github.io/Adeversarial-Speech-with-YourTTS.



## **30. Unraveling Attacks in Machine Learning-based IoT Ecosystems: A Survey and the Open Libraries Behind Them**

cs.CR

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.11723v1) [paper-pdf](http://arxiv.org/pdf/2401.11723v1)

**Authors**: Chao Liu, Boxi Chen, Wei Shao, Chris Zhang, Kelvin Wong, Yi Zhang

**Abstract**: The advent of the Internet of Things (IoT) has brought forth an era of unprecedented connectivity, with an estimated 80 billion smart devices expected to be in operation by the end of 2025. These devices facilitate a multitude of smart applications, enhancing the quality of life and efficiency across various domains. Machine Learning (ML) serves as a crucial technology, not only for analyzing IoT-generated data but also for diverse applications within the IoT ecosystem. For instance, ML finds utility in IoT device recognition, anomaly detection, and even in uncovering malicious activities. This paper embarks on a comprehensive exploration of the security threats arising from ML's integration into various facets of IoT, spanning various attack types including membership inference, adversarial evasion, reconstruction, property inference, model extraction, and poisoning attacks. Unlike previous studies, our work offers a holistic perspective, categorizing threats based on criteria such as adversary models, attack targets, and key security attributes (confidentiality, availability, and integrity). We delve into the underlying techniques of ML attacks in IoT environment, providing a critical evaluation of their mechanisms and impacts. Furthermore, our research thoroughly assesses 65 libraries, both author-contributed and third-party, evaluating their role in safeguarding model and data privacy. We emphasize the availability and usability of these libraries, aiming to arm the community with the necessary tools to bolster their defenses against the evolving threat landscape. Through our comprehensive review and analysis, this paper seeks to contribute to the ongoing discourse on ML-based IoT security, offering valuable insights and practical solutions to secure ML models and data in the rapidly expanding field of artificial intelligence in IoT.



## **31. HashVFL: Defending Against Data Reconstruction Attacks in Vertical Federated Learning**

cs.CR

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2212.00325v2) [paper-pdf](http://arxiv.org/pdf/2212.00325v2)

**Authors**: Pengyu Qiu, Xuhong Zhang, Shouling Ji, Chong Fu, Xing Yang, Ting Wang

**Abstract**: Vertical Federated Learning (VFL) is a trending collaborative machine learning model training solution. Existing industrial frameworks employ secure multi-party computation techniques such as homomorphic encryption to ensure data security and privacy. Despite these efforts, studies have revealed that data leakage remains a risk in VFL due to the correlations between intermediate representations and raw data. Neural networks can accurately capture these correlations, allowing an adversary to reconstruct the data. This emphasizes the need for continued research into securing VFL systems.   Our work shows that hashing is a promising solution to counter data reconstruction attacks. The one-way nature of hashing makes it difficult for an adversary to recover data from hash codes. However, implementing hashing in VFL presents new challenges, including vanishing gradients and information loss. To address these issues, we propose HashVFL, which integrates hashing and simultaneously achieves learnability, bit balance, and consistency.   Experimental results indicate that HashVFL effectively maintains task performance while defending against data reconstruction attacks. It also brings additional benefits in reducing the degree of label leakage, mitigating adversarial attacks, and detecting abnormal inputs. We hope our work will inspire further research into the potential applications of HashVFL.



## **32. LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate**

cs.LG

AAAI 2024 main track. Code available on Github (see abstract).  Appendix is included in this updated version

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2312.13118v2) [paper-pdf](http://arxiv.org/pdf/2312.13118v2)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking \emph{given} pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose {\em Lipschitz Regularized Surrogate} (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.



## **33. Analyzing the Quality Attributes of AI Vision Models in Open Repositories Under Adversarial Attacks**

cs.CR

10 pages

**SubmitDate**: 2024-01-22    [abs](http://arxiv.org/abs/2401.12261v1) [paper-pdf](http://arxiv.org/pdf/2401.12261v1)

**Authors**: Zerui Wang, Yan Liu

**Abstract**: As AI models rapidly evolve, they are frequently released to open repositories, such as HuggingFace. It is essential to perform quality assurance validation on these models before integrating them into the production development lifecycle. In addition to evaluating efficiency in terms of balanced accuracy and computing costs, adversarial attacks are potential threats to the robustness and explainability of AI models. Meanwhile, XAI applies algorithms that approximate inputs to outputs post-hoc to identify the contributing features. Adversarial perturbations may also degrade the utility of XAI explanations that require further investigation. In this paper, we present an integrated process designed for downstream evaluation tasks, including validating AI model accuracy, evaluating robustness with benchmark perturbations, comparing explanation utility, and assessing overhead. We demonstrate an evaluation scenario involving six computer vision models, which include CNN-based, Transformer-based, and hybrid architectures, three types of perturbations, and five XAI methods, resulting in ninety unique combinations. The process reveals the explanation utility among the XAI methods in terms of the identified key areas responding to the adversarial perturbation. The process produces aggregated results that illustrate multiple attributes of each AI model.



## **34. Reducing Usefulness of Stolen Credentials in SSO Contexts**

cs.CR

8 pages, 5 figures

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11599v1) [paper-pdf](http://arxiv.org/pdf/2401.11599v1)

**Authors**: Sam Hays, Michael Sandborn, Dr. Jules White

**Abstract**: Approximately 61% of cyber attacks involve adversaries in possession of valid credentials. Attackers acquire credentials through various means, including phishing, dark web data drops, password reuse, etc. Multi-factor authentication (MFA) helps to thwart attacks that use valid credentials, but attackers still commonly breach systems by tricking users into accepting MFA step up requests through techniques, such as ``MFA Bombing'', where multiple requests are sent to a user until they accept one. Currently, there are several solutions to this problem, each with varying levels of security and increasing invasiveness on user devices. This paper proposes a token-based enrollment architecture that is less invasive to user devices than mobile device management, but still offers strong protection against use of stolen credentials and MFA attacks.



## **35. Thundernna: a white box adversarial attack**

cs.LG

10 pages, 5 figures

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2111.12305v2) [paper-pdf](http://arxiv.org/pdf/2111.12305v2)

**Authors**: Linfeng Ye, Shayan Mohajer Hamidi

**Abstract**: The existing work shows that the neural network trained by naive gradient-based optimization method is prone to adversarial attacks, adds small malicious on the ordinary input is enough to make the neural network wrong. At the same time, the attack against a neural network is the key to improving its robustness. The training against adversarial examples can make neural networks resist some kinds of adversarial attacks. At the same time, the adversarial attack against a neural network can also reveal some characteristics of the neural network, a complex high-dimensional non-linear function, as discussed in previous work.   In This project, we develop a first-order method to attack the neural network. Compare with other first-order attacks, our method has a much higher success rate. Furthermore, it is much faster than second-order attacks and multi-steps first-order attacks.



## **36. How Robust Are Energy-Based Models Trained With Equilibrium Propagation?**

cs.LG

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11543v1) [paper-pdf](http://arxiv.org/pdf/2401.11543v1)

**Authors**: Siddharth Mansingh, Michal Kucer, Garrett Kenyon, Juston Moore, Michael Teti

**Abstract**: Deep neural networks (DNNs) are easily fooled by adversarial perturbations that are imperceptible to humans. Adversarial training, a process where adversarial examples are added to the training set, is the current state-of-the-art defense against adversarial attacks, but it lowers the model's accuracy on clean inputs, is computationally expensive, and offers less robustness to natural noise. In contrast, energy-based models (EBMs), which were designed for efficient implementation in neuromorphic hardware and physical systems, incorporate feedback connections from each layer to the previous layer, yielding a recurrent, deep-attractor architecture which we hypothesize should make them naturally robust. Our work is the first to explore the robustness of EBMs to both natural corruptions and adversarial attacks, which we do using the CIFAR-10 and CIFAR-100 datasets. We demonstrate that EBMs are more robust than transformers and display comparable robustness to adversarially-trained DNNs on gradient-based (white-box) attacks, query-based (black-box) attacks, and natural perturbations without sacrificing clean accuracy, and without the need for adversarial training or additional training techniques.



## **37. Finding a Needle in the Adversarial Haystack: A Targeted Paraphrasing Approach For Uncovering Edge Cases with Minimal Distribution Distortion**

cs.CL

EACL 2024 - Main conference

**SubmitDate**: 2024-01-21    [abs](http://arxiv.org/abs/2401.11373v1) [paper-pdf](http://arxiv.org/pdf/2401.11373v1)

**Authors**: Aly M. Kassem, Sherif Saad

**Abstract**: Adversarial attacks against NLP Deep Learning models are a significant concern. In particular, adversarial samples exploit the model's sensitivity to small input changes. While these changes appear insignificant on the semantics of the input sample, they result in significant decay in model performance. In this paper, we propose Targeted Paraphrasing via RL (TPRL), an approach to automatically learn a policy to generate challenging samples that most likely improve the model's performance. TPRL leverages FLAN T5, a language model, as a generator and employs a self learned policy using a proximal policy gradient to generate the adversarial examples automatically. TPRL's reward is based on the confusion induced in the classifier, preserving the original text meaning through a Mutual Implication score. We demonstrate and evaluate TPRL's effectiveness in discovering natural adversarial attacks and improving model performance through extensive experiments on four diverse NLP classification tasks via Automatic and Human evaluation. TPRL outperforms strong baselines, exhibits generalizability across classifiers and datasets, and combines the strengths of language modeling and reinforcement learning to generate diverse and influential adversarial examples.



## **38. Robustness Against Adversarial Attacks via Learning Confined Adversarial Polytopes**

cs.LG

The paper has been accepted in ICASSP 2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.07991v2) [paper-pdf](http://arxiv.org/pdf/2401.07991v2)

**Authors**: Shayan Mohajer Hamidi, Linfeng Ye

**Abstract**: Deep neural networks (DNNs) could be deceived by generating human-imperceptible perturbations of clean samples. Therefore, enhancing the robustness of DNNs against adversarial attacks is a crucial task. In this paper, we aim to train robust DNNs by limiting the set of outputs reachable via a norm-bounded perturbation added to a clean sample. We refer to this set as adversarial polytope, and each clean sample has a respective adversarial polytope. Indeed, if the respective polytopes for all the samples are compact such that they do not intersect the decision boundaries of the DNN, then the DNN is robust against adversarial samples. Hence, the inner-working of our algorithm is based on learning \textbf{c}onfined \textbf{a}dversarial \textbf{p}olytopes (CAP). By conducting a thorough set of experiments, we demonstrate the effectiveness of CAP over existing adversarial robustness methods in improving the robustness of models against state-of-the-art attacks including AutoAttack.



## **39. Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts**

cs.CR

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2311.09127v2) [paper-pdf](http://arxiv.org/pdf/2311.09127v2)

**Authors**: Yuanwei Wu, Xiang Li, Yixin Liu, Pan Zhou, Lichao Sun

**Abstract**: Existing work on jailbreak Multimodal Large Language Models (MLLMs) has focused primarily on adversarial examples in model inputs, with less attention to vulnerabilities, especially in model API. To fill the research gap, we carry out the following work: 1) We discover a system prompt leakage vulnerability in GPT-4V. Through carefully designed dialogue, we successfully extract the internal system prompts of GPT-4V. This finding indicates potential exploitable security risks in MLLMs; 2) Based on the acquired system prompts, we propose a novel MLLM jailbreaking attack method termed SASP (Self-Adversarial Attack via System Prompt). By employing GPT-4 as a red teaming tool against itself, we aim to search for potential jailbreak prompts leveraging stolen system prompts. Furthermore, in pursuit of better performance, we also add human modification based on GPT-4's analysis, which further improves the attack success rate to 98.7\%; 3) We evaluated the effect of modifying system prompts to defend against jailbreaking attacks. Results show that appropriately designed system prompts can significantly reduce jailbreak success rates. Overall, our work provides new insights into enhancing MLLM security, demonstrating the important role of system prompts in jailbreaking. This finding could be leveraged to greatly facilitate jailbreak success rates while also holding the potential for defending against jailbreaks.



## **40. Susceptibility of Adversarial Attack on Medical Image Segmentation Models**

eess.IV

6 pages, 8 figures, presented at 2023 IEEE 20th International  Symposium on Biomedical Imaging (ISBI) conference

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11224v1) [paper-pdf](http://arxiv.org/pdf/2401.11224v1)

**Authors**: Zhongxuan Wang, Leo Xu

**Abstract**: The nature of deep neural networks has given rise to a variety of attacks, but little work has been done to address the effect of adversarial attacks on segmentation models trained on MRI datasets. In light of the grave consequences that such attacks could cause, we explore four models from the U-Net family and examine their responses to the Fast Gradient Sign Method (FGSM) attack. We conduct FGSM attacks on each of them and experiment with various schemes to conduct the attacks. In this paper, we find that medical imaging segmentation models are indeed vulnerable to adversarial attacks and that there is a negligible correlation between parameter size and adversarial attack success. Furthermore, we show that using a different loss function than the one used for training yields higher adversarial attack success, contrary to what the FGSM authors suggested. In future efforts, we will conduct the experiments detailed in this paper with more segmentation models and different attacks. We will also attempt to find ways to counteract the attacks by using model ensembles or special data augmentations. Our code is available at https://github.com/ZhongxuanWang/adv_attk



## **41. Generalizing Speaker Verification for Spoof Awareness in the Embedding Space**

cs.CR

To appear in IEEE/ACM Transactions on Audio, Speech, and Language  Processing

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11156v1) [paper-pdf](http://arxiv.org/pdf/2401.11156v1)

**Authors**: Xuechen Liu, Md Sahidullah, Kong Aik Lee, Tomi Kinnunen

**Abstract**: It is now well-known that automatic speaker verification (ASV) systems can be spoofed using various types of adversaries. The usual approach to counteract ASV systems against such attacks is to develop a separate spoofing countermeasure (CM) module to classify speech input either as a bonafide, or a spoofed utterance. Nevertheless, such a design requires additional computation and utilization efforts at the authentication stage. An alternative strategy involves a single monolithic ASV system designed to handle both zero-effort imposter (non-targets) and spoofing attacks. Such spoof-aware ASV systems have the potential to provide stronger protections and more economic computations. To this end, we propose to generalize the standalone ASV (G-SASV) against spoofing attacks, where we leverage limited training data from CM to enhance a simple backend in the embedding space, without the involvement of a separate CM module during the test (authentication) phase. We propose a novel yet simple backend classifier based on deep neural networks and conduct the study via domain adaptation and multi-task integration of spoof embeddings at the training stage. Experiments are conducted on the ASVspoof 2019 logical access dataset, where we improve the performance of statistical ASV backends on the joint (bonafide and spoofed) and spoofed conditions by a maximum of 36.2% and 49.8% in terms of equal error rates, respectively.



## **42. CARE: Ensemble Adversarial Robustness Evaluation Against Adaptive Attackers for Security Applications**

cs.CR

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2401.11126v1) [paper-pdf](http://arxiv.org/pdf/2401.11126v1)

**Authors**: Hangsheng Zhang, Jiqiang Liu, Jinsong Dong

**Abstract**: Ensemble defenses, are widely employed in various security-related applications to enhance model performance and robustness. The widespread adoption of these techniques also raises many questions: Are general ensembles defenses guaranteed to be more robust than individuals? Will stronger adaptive attacks defeat existing ensemble defense strategies as the cybersecurity arms race progresses? Can ensemble defenses achieve adversarial robustness to different types of attacks simultaneously and resist the continually adjusted adaptive attacks? Unfortunately, these critical questions remain unresolved as there are no platforms for comprehensive evaluation of ensemble adversarial attacks and defenses in the cybersecurity domain. In this paper, we propose a general Cybersecurity Adversarial Robustness Evaluation (CARE) platform aiming to bridge this gap.



## **43. Universal Backdoor Attacks**

cs.LG

Accepted for publication at ICLR 2024

**SubmitDate**: 2024-01-20    [abs](http://arxiv.org/abs/2312.00157v2) [paper-pdf](http://arxiv.org/pdf/2312.00157v2)

**Authors**: Benjamin Schneider, Nils Lukas, Florian Kerschbaum

**Abstract**: Web-scraped datasets are vulnerable to data poisoning, which can be used for backdooring deep image classifiers during training. Since training on large datasets is expensive, a model is trained once and re-used many times. Unlike adversarial examples, backdoor attacks often target specific classes rather than any class learned by the model. One might expect that targeting many classes through a naive composition of attacks vastly increases the number of poison samples. We show this is not necessarily true and more efficient, universal data poisoning attacks exist that allow controlling misclassifications from any source class into any target class with a small increase in poison samples. Our idea is to generate triggers with salient characteristics that the model can learn. The triggers we craft exploit a phenomenon we call inter-class poison transferability, where learning a trigger from one class makes the model more vulnerable to learning triggers for other classes. We demonstrate the effectiveness and robustness of our universal backdoor attacks by controlling models with up to 6,000 classes while poisoning only 0.15% of the training dataset. Our source code is available at https://github.com/Ben-Schneider-code/Universal-Backdoor-Attacks.



## **44. Ensembler: Combating model inversion attacks using model ensemble during collaborative inference**

cs.CR

in submission

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10859v1) [paper-pdf](http://arxiv.org/pdf/2401.10859v1)

**Authors**: Dancheng Liu, Jinjun Xiong

**Abstract**: Deep learning models have exhibited remarkable performance across various domains. Nevertheless, the burgeoning model sizes compel edge devices to offload a significant portion of the inference process to the cloud. While this practice offers numerous advantages, it also raises critical concerns regarding user data privacy. In scenarios where the cloud server's trustworthiness is in question, the need for a practical and adaptable method to safeguard data privacy becomes imperative. In this paper, we introduce Ensembler, an extensible framework designed to substantially increase the difficulty of conducting model inversion attacks for adversarial parties. Ensembler leverages model ensembling on the adversarial server, running in parallel with existing approaches that introduce perturbations to sensitive data during colloborative inference. Our experiments demonstrate that when combined with even basic Gaussian noise, Ensembler can effectively shield images from reconstruction attacks, achieving recognition levels that fall below human performance in some strict settings, significantly outperforming baseline methods lacking the Ensembler framework.



## **45. Privacy-Preserving Neural Graph Databases**

cs.DB

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2312.15591v2) [paper-pdf](http://arxiv.org/pdf/2312.15591v2)

**Authors**: Qi Hu, Haoran Li, Jiaxin Bai, Yangqiu Song

**Abstract**: In the era of big data and rapidly evolving information systems, efficient and accurate data retrieval has become increasingly crucial. Neural graph databases (NGDBs) have emerged as a powerful paradigm that combines the strengths of graph databases (graph DBs) and neural networks to enable efficient storage, retrieval, and analysis of graph-structured data. The usage of neural embedding storage and complex neural logical query answering provides NGDBs with generalization ability. When the graph is incomplete, by extracting latent patterns and representations, neural graph databases can fill gaps in the graph structure, revealing hidden relationships and enabling accurate query answering. Nevertheless, this capability comes with inherent trade-offs, as it introduces additional privacy risks to the database. Malicious attackers can infer more sensitive information in the database using well-designed combinatorial queries, such as by comparing the answer sets of where Turing Award winners born before 1950 and after 1940 lived, the living places of Turing Award winner Hinton are probably exposed, although the living places may have been deleted in the training due to the privacy concerns. In this work, inspired by the privacy protection in graph embeddings, we propose a privacy-preserving neural graph database (P-NGDB) to alleviate the risks of privacy leakage in NGDBs. We introduce adversarial training techniques in the training stage to force the NGDBs to generate indistinguishable answers when queried with private information, enhancing the difficulty of inferring sensitive information through combinations of multiple innocuous queries. Extensive experiment results on three datasets show that P-NGDB can effectively protect private information in the graph database while delivering high-quality public answers responses to queries.



## **46. Explainable and Transferable Adversarial Attack for ML-Based Network Intrusion Detectors**

cs.CR

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10691v1) [paper-pdf](http://arxiv.org/pdf/2401.10691v1)

**Authors**: Hangsheng Zhang, Dongqi Han, Yinlong Liu, Zhiliang Wang, Jiyan Sun, Shangyuan Zhuang, Jiqiang Liu, Jinsong Dong

**Abstract**: espite being widely used in network intrusion detection systems (NIDSs), machine learning (ML) has proven to be highly vulnerable to adversarial attacks. White-box and black-box adversarial attacks of NIDS have been explored in several studies. However, white-box attacks unrealistically assume that the attackers have full knowledge of the target NIDSs. Meanwhile, existing black-box attacks can not achieve high attack success rate due to the weak adversarial transferability between models (e.g., neural networks and tree models). Additionally, neither of them explains why adversarial examples exist and why they can transfer across models. To address these challenges, this paper introduces ETA, an Explainable Transfer-based Black-Box Adversarial Attack framework. ETA aims to achieve two primary objectives: 1) create transferable adversarial examples applicable to various ML models and 2) provide insights into the existence of adversarial examples and their transferability within NIDSs. Specifically, we first provide a general transfer-based adversarial attack method applicable across the entire ML space. Following that, we exploit a unique insight based on cooperative game theory and perturbation interpretations to explain adversarial examples and adversarial transferability. On this basis, we propose an Important-Sensitive Feature Selection (ISFS) method to guide the search for adversarial examples, achieving stronger transferability and ensuring traffic-space constraints.



## **47. FIMBA: Evaluating the Robustness of AI in Genomics via Feature Importance Adversarial Attacks**

cs.LG

15 pages, core code available at:  https://github.com/HeorhiiS/fimba-attack

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10657v1) [paper-pdf](http://arxiv.org/pdf/2401.10657v1)

**Authors**: Heorhii Skovorodnikov, Hoda Alkhzaimi

**Abstract**: With the steady rise of the use of AI in bio-technical applications and the widespread adoption of genomics sequencing, an increasing amount of AI-based algorithms and tools is entering the research and production stage affecting critical decision-making streams like drug discovery and clinical outcomes. This paper demonstrates the vulnerability of AI models often utilized downstream tasks on recognized public genomics datasets. We undermine model robustness by deploying an attack that focuses on input transformation while mimicking the real data and confusing the model decision-making, ultimately yielding a pronounced deterioration in model performance. Further, we enhance our approach by generating poisoned data using a variational autoencoder-based model. Our empirical findings unequivocally demonstrate a decline in model performance, underscored by diminished accuracy and an upswing in false positives and false negatives. Furthermore, we analyze the resulting adversarial samples via spectral analysis yielding conclusions for countermeasures against such attacks.



## **48. Adversarially Robust Signed Graph Contrastive Learning from Balance Augmentation**

cs.LG

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10590v1) [paper-pdf](http://arxiv.org/pdf/2401.10590v1)

**Authors**: Jialong Zhou, Xing Ai, Yuni Lai, Kai Zhou

**Abstract**: Signed graphs consist of edges and signs, which can be separated into structural information and balance-related information, respectively. Existing signed graph neural networks (SGNNs) typically rely on balance-related information to generate embeddings. Nevertheless, the emergence of recent adversarial attacks has had a detrimental impact on the balance-related information. Similar to how structure learning can restore unsigned graphs, balance learning can be applied to signed graphs by improving the balance degree of the poisoned graph. However, this approach encounters the challenge "Irreversibility of Balance-related Information" - while the balance degree improves, the restored edges may not be the ones originally affected by attacks, resulting in poor defense effectiveness. To address this challenge, we propose a robust SGNN framework called Balance Augmented-Signed Graph Contrastive Learning (BA-SGCL), which combines Graph Contrastive Learning principles with balance augmentation techniques. Experimental results demonstrate that BA-SGCL not only enhances robustness against existing adversarial attacks but also achieves superior performance on link sign prediction task across various datasets.



## **49. PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks**

cs.CR

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.10586v1) [paper-pdf](http://arxiv.org/pdf/2401.10586v1)

**Authors**: Ping Guo, Zhiyuan Yang, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: Black-box query-based attacks constitute significant threats to Machine Learning as a Service (MLaaS) systems since they can generate adversarial examples without accessing the target model's architecture and parameters. Traditional defense mechanisms, such as adversarial training, gradient masking, and input transformations, either impose substantial computational costs or compromise the test accuracy of non-adversarial inputs. To address these challenges, we propose an efficient defense mechanism, PuriDefense, that employs random patch-wise purifications with an ensemble of lightweight purification models at a low level of inference cost. These models leverage the local implicit function and rebuild the natural image manifold. Our theoretical analysis suggests that this approach slows down the convergence of query-based attacks by incorporating randomness into purifications. Extensive experiments on CIFAR-10 and ImageNet validate the effectiveness of our proposed purifier-based defense mechanism, demonstrating significant improvements in robustness against query-based attacks.



## **50. Hijacking Attacks against Neural Networks by Analyzing Training Data**

cs.CR

Full version with major polishing, compared to the Usenix Security  2024 edition

**SubmitDate**: 2024-01-19    [abs](http://arxiv.org/abs/2401.09740v2) [paper-pdf](http://arxiv.org/pdf/2401.09740v2)

**Authors**: Yunjie Ge, Qian Wang, Huayang Huang, Qi Li, Cong Wang, Chao Shen, Lingchen Zhao, Peipei Jiang, Zheng Fang, Shenyi Zhang

**Abstract**: Backdoors and adversarial examples are the two primary threats currently faced by deep neural networks (DNNs). Both attacks attempt to hijack the model behaviors with unintended outputs by introducing (small) perturbations to the inputs. Backdoor attacks, despite the high success rates, often require a strong assumption, which is not always easy to achieve in reality. Adversarial example attacks, which put relatively weaker assumptions on attackers, often demand high computational resources, yet do not always yield satisfactory success rates when attacking mainstream black-box models in the real world. These limitations motivate the following research question: can model hijacking be achieved more simply, with a higher attack success rate and more reasonable assumptions? In this paper, we propose CleanSheet, a new model hijacking attack that obtains the high performance of backdoor attacks without requiring the adversary to tamper with the model training process. CleanSheet exploits vulnerabilities in DNNs stemming from the training data. Specifically, our key idea is to treat part of the clean training data of the target model as "poisoned data," and capture the characteristics of these data that are more sensitive to the model (typically called robust features) to construct "triggers." These triggers can be added to any input example to mislead the target model, similar to backdoor attacks. We validate the effectiveness of CleanSheet through extensive experiments on 5 datasets, 79 normally trained models, 68 pruned models, and 39 defensive models. Results show that CleanSheet exhibits performance comparable to state-of-the-art backdoor attacks, achieving an average attack success rate (ASR) of 97.5% on CIFAR-100 and 92.4% on GTSRB, respectively. Furthermore, CleanSheet consistently maintains a high ASR, when confronted with various mainstream backdoor defenses.



