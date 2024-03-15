# Latest Adversarial Attack Papers
**update at 2024-03-15 11:36:56**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Expressive Losses for Verified Robustness via Convex Combinations**

cs.LG

ICLR 2024

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2305.13991v2) [paper-pdf](http://arxiv.org/pdf/2305.13991v2)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth, Alessio Lomuscio

**Abstract**: In order to train networks for verified adversarial robustness, it is common to over-approximate the worst-case loss over perturbation regions, resulting in networks that attain verifiability at the expense of standard performance. As shown in recent work, better trade-offs between accuracy and robustness can be obtained by carefully coupling adversarial training with over-approximations. We hypothesize that the expressivity of a loss function, which we formalize as the ability to span a range of trade-offs between lower and upper bounds to the worst-case loss through a single parameter (the over-approximation coefficient), is key to attaining state-of-the-art performance. To support our hypothesis, we show that trivial expressive losses, obtained via convex combinations between adversarial attacks and IBP bounds, yield state-of-the-art results across a variety of settings in spite of their conceptual simplicity. We provide a detailed analysis of the relationship between the over-approximation coefficient and performance profiles across different expressive losses, showing that, while expressivity is essential, better approximations of the worst-case loss are not necessarily linked to superior robustness-accuracy trade-offs.



## **2. What Sketch Explainability Really Means for Downstream Tasks**

cs.CV

CVPR 2024

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09480v1) [paper-pdf](http://arxiv.org/pdf/2403.09480v1)

**Authors**: Hmrishav Bandyopadhyay, Pinaki Nath Chowdhury, Ayan Kumar Bhunia, Aneeshan Sain, Tao Xiang, Yi-Zhe Song

**Abstract**: In this paper, we explore the unique modality of sketch for explainability, emphasising the profound impact of human strokes compared to conventional pixel-oriented studies. Beyond explanations of network behavior, we discern the genuine implications of explainability across diverse downstream sketch-related tasks. We propose a lightweight and portable explainability solution -- a seamless plugin that integrates effortlessly with any pre-trained model, eliminating the need for re-training. Demonstrating its adaptability, we present four applications: highly studied retrieval and generation, and completely novel assisted drawing and sketch adversarial attacks. The centrepiece to our solution is a stroke-level attribution map that takes different forms when linked with downstream tasks. By addressing the inherent non-differentiability of rasterisation, we enable explanations at both coarse stroke level (SLA) and partial stroke level (P-SLA), each with its advantages for specific downstream tasks.



## **3. Adversarial Fine-tuning of Compressed Neural Networks for Joint Improvement of Robustness and Efficiency**

cs.LG

22 pages, 4 figures, 6 tables

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09441v1) [paper-pdf](http://arxiv.org/pdf/2403.09441v1)

**Authors**: Hallgrimur Thorsteinsson, Valdemar J Henriksen, Tong Chen, Raghavendra Selvan

**Abstract**: As deep learning (DL) models are increasingly being integrated into our everyday lives, ensuring their safety by making them robust against adversarial attacks has become increasingly critical. DL models have been found to be susceptible to adversarial attacks which can be achieved by introducing small, targeted perturbations to disrupt the input data. Adversarial training has been presented as a mitigation strategy which can result in more robust models. This adversarial robustness comes with additional computational costs required to design adversarial attacks during training. The two objectives -- adversarial robustness and computational efficiency -- then appear to be in conflict of each other. In this work, we explore the effects of two different model compression methods -- structured weight pruning and quantization -- on adversarial robustness. We specifically explore the effects of fine-tuning on compressed models, and present the trade-off between standard fine-tuning and adversarial fine-tuning. Our results show that compression does not inherently lead to loss in model robustness and adversarial fine-tuning of a compressed model can yield large improvement to the robustness performance of models. We present experiments on two benchmark datasets showing that adversarial fine-tuning of compressed models can achieve robustness performance comparable to adversarially trained models, while also improving computational efficiency.



## **4. Divide-and-Conquer Attack: Harnessing the Power of LLM to Bypass Safety Filters of Text-to-Image Models**

cs.AI

23 pages, 11 figures, under review

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2312.07130v3) [paper-pdf](http://arxiv.org/pdf/2312.07130v3)

**Authors**: Yimo Deng, Huangxun Chen

**Abstract**: Text-to-image (TTI) models offer many innovative services but also raise ethical concerns due to their potential to generate unethical images. Most public TTI services employ safety filters to prevent unintended images. In this work, we introduce the Divide-and-Conquer Attack to circumvent the safety filters of state-of the-art TTI models, including DALL-E 3 and Midjourney. Our attack leverages LLMs as text transformation agents to create adversarial prompts. We design attack helper prompts that effectively guide LLMs to break down an unethical drawing intent into multiple benign descriptions of individual image elements, allowing them to bypass safety filters while still generating unethical images. Because the latent harmful meaning only becomes apparent when all individual elements are drawn together. Our evaluation demonstrates that our attack successfully circumvents multiple strong closed-box safety filters. The comprehensive success rate of DACA bypassing the safety filters of the state-of-the-art TTI engine DALL-E 3 is above 85%, while the success rate for bypassing Midjourney V6 exceeds 75%. Our findings have more severe security implications than methods of manual crafting or iterative TTI model querying due to lower attack barrier, enhanced interpretability , and better adaptation to defense. Our prototype is available at: https://github.com/researchcode001/Divide-and-Conquer-Attack



## **5. AVIBench: Towards Evaluating the Robustness of Large Vision-Language Model on Adversarial Visual-Instructions**

cs.CV

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09346v1) [paper-pdf](http://arxiv.org/pdf/2403.09346v1)

**Authors**: Hao Zhang, Wenqi Shao, Hong Liu, Yongqiang Ma, Ping Luo, Yu Qiao, Kaipeng Zhang

**Abstract**: Large Vision-Language Models (LVLMs) have shown significant progress in well responding to visual-instructions from users. However, these instructions, encompassing images and text, are susceptible to both intentional and inadvertent attacks. Despite the critical importance of LVLMs' robustness against such threats, current research in this area remains limited. To bridge this gap, we introduce AVIBench, a framework designed to analyze the robustness of LVLMs when facing various adversarial visual-instructions (AVIs), including four types of image-based AVIs, ten types of text-based AVIs, and nine types of content bias AVIs (such as gender, violence, cultural, and racial biases, among others). We generate 260K AVIs encompassing five categories of multimodal capabilities (nine tasks) and content bias. We then conduct a comprehensive evaluation involving 14 open-source LVLMs to assess their performance. AVIBench also serves as a convenient tool for practitioners to evaluate the robustness of LVLMs against AVIs. Our findings and extensive experimental results shed light on the vulnerabilities of LVLMs, and highlight that inherent biases exist even in advanced closed-source LVLMs like GeminiProVision and GPT-4V. This underscores the importance of enhancing the robustness, security, and fairness of LVLMs. The source code and benchmark will be made publicly available.



## **6. Privacy Preserving Anomaly Detection on Homomorphic Encrypted Data from IoT Sensors**

cs.CR

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09322v1) [paper-pdf](http://arxiv.org/pdf/2403.09322v1)

**Authors**: Anca Hangan, Dragos Lazea, Tudor Cioara

**Abstract**: IoT devices have become indispensable components of our lives, and the advancement of AI technologies will make them even more pervasive, increasing the vulnerability to malfunctions or cyberattacks and raising privacy concerns. Encryption can mitigate these challenges; however, most existing anomaly detection techniques decrypt the data to perform the analysis, potentially undermining the encryption protection provided during transit or storage. Homomorphic encryption schemes are promising solutions as they enable the processing and execution of operations on IoT data while still encrypted, however, these schemes offer only limited operations, which poses challenges to their practical usage. In this paper, we propose a novel privacy-preserving anomaly detection solution designed for homomorphically encrypted data generated by IoT devices that efficiently detects abnormal values without performing decryption. We have adapted the Histogram-based anomaly detection technique for TFHE scheme to address limitations related to the input size and the depth of computation by implementing vectorized support operations. These operations include addition, value placement in buckets, labeling abnormal buckets based on a threshold frequency, labeling abnormal values based on their range, and bucket labels. Evaluation results show that the solution effectively detects anomalies without requiring data decryption and achieves consistent results comparable to the mechanism operating on plain data. Also, it shows robustness and resilience against various challenges commonly encountered in IoT environments, such as noisy sensor data, adversarial attacks, communication failures, and device malfunctions. Moreover, the time and computational overheads determined for several solution configurations, despite being large, are reasonable compared to those reported in existing literature.



## **7. X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network**

cs.CR

This is the Accepted version of an article for publication in IEEE  TVT

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2303.12278v3) [paper-pdf](http://arxiv.org/pdf/2303.12278v3)

**Authors**: Seonghoon Jeong, Sangho Lee, Hwejae Lee, Huy Kang Kim

**Abstract**: Controller Area Network (CAN) is an essential networking protocol that connects multiple electronic control units (ECUs) in a vehicle. However, CAN-based in-vehicle networks (IVNs) face security risks owing to the CAN mechanisms. An adversary can sabotage a vehicle by leveraging the security risks if they can access the CAN bus. Thus, recent actions and cybersecurity regulations (e.g., UNR 155) require carmakers to implement intrusion detection systems (IDSs) in their vehicles. The IDS should detect cyberattacks and provide additional information to analyze conducted attacks. Although many IDSs have been proposed, considerations regarding their feasibility and explainability remain lacking. This study proposes X-CANIDS, which is a novel IDS for CAN-based IVNs. X-CANIDS dissects the payloads in CAN messages into human-understandable signals using a CAN database. The signals improve the intrusion detection performance compared with the use of bit representations of raw payloads. These signals also enable an understanding of which signal or ECU is under attack. X-CANIDS can detect zero-day attacks because it does not require any labeled dataset in the training phase. We confirmed the feasibility of the proposed method through a benchmark test on an automotive-grade embedded device with a GPU. The results of this work will be valuable to carmakers and researchers considering the installation of in-vehicle IDSs for their vehicles.



## **8. Soften to Defend: Towards Adversarial Robustness via Self-Guided Label Refinement**

cs.LG

Accepted to CVPR 2024

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09101v1) [paper-pdf](http://arxiv.org/pdf/2403.09101v1)

**Authors**: Daiwei Yu, Zhuorong Li, Lina Wei, Canghong Jin, Yun Zhang, Sixian Chan

**Abstract**: Adversarial training (AT) is currently one of the most effective ways to obtain the robustness of deep neural networks against adversarial attacks. However, most AT methods suffer from robust overfitting, i.e., a significant generalization gap in adversarial robustness between the training and testing curves. In this paper, we first identify a connection between robust overfitting and the excessive memorization of noisy labels in AT from a view of gradient norm. As such label noise is mainly caused by a distribution mismatch and improper label assignments, we are motivated to propose a label refinement approach for AT. Specifically, our Self-Guided Label Refinement first self-refines a more accurate and informative label distribution from over-confident hard labels, and then it calibrates the training by dynamically incorporating knowledge from self-distilled models into the current model and thus requiring no external teachers. Empirical results demonstrate that our method can simultaneously boost the standard accuracy and robust performance across multiple benchmark datasets, attack types, and architectures. In addition, we also provide a set of analyses from the perspectives of information theory to dive into our method and suggest the importance of soft labels for robust generalization.



## **9. Chaotic Masking Protocol for Secure Communication and Attack Detection in Remote Estimation of Cyber-Physical Systems**

eess.SY

8 pages, 7 figures

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.09076v1) [paper-pdf](http://arxiv.org/pdf/2403.09076v1)

**Authors**: Tao Chen, Andreu Cecilia, Daniele Astolfi, Lei Wang, Zhitao Liu, Hongye Su

**Abstract**: In remote estimation of cyber-physical systems (CPSs), sensor measurements transmitted through network may be attacked by adversaries, leading to leakage risk of privacy (e.g., the system state), and/or failure of the remote estimator. To deal with this problem, a chaotic masking protocol is proposed in this paper to secure the sensor measurements transmission. In detail, at the plant side, a chaotic dynamic system is deployed to encode the sensor measurement, and at the estimator side, an estimator estimates both states of the physical plant and the chaotic system. With this protocol, no additional secure communication links is needed for synchronization, and the masking effect can be perfectly removed when the estimator is in steady state. Furthermore, this masking protocol can deal with multiple types of attacks, i.e., eavesdropping attack, replay attack, and stealthy false data injection attack.



## **10. Attacking the Diebold Signature Variant -- RSA Signatures with Unverified High-order Padding**

cs.CR

**SubmitDate**: 2024-03-14    [abs](http://arxiv.org/abs/2403.01048v2) [paper-pdf](http://arxiv.org/pdf/2403.01048v2)

**Authors**: Ryan W. Gardner, Tadayoshi Kohno, Alec Yasinsac

**Abstract**: We examine a natural but improper implementation of RSA signature verification deployed on the widely used Diebold Touch Screen and Optical Scan voting machines. In the implemented scheme, the verifier fails to examine a large number of the high-order bits of signature padding and the public exponent is three. We present an very mathematically simple attack that enables an adversary to forge signatures on arbitrary messages in a negligible amount of time.



## **11. Verifix: Post-Training Correction to Improve Label Noise Robustness with Verified Samples**

cs.LG

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08618v1) [paper-pdf](http://arxiv.org/pdf/2403.08618v1)

**Authors**: Sangamesh Kodge, Deepak Ravikumar, Gobinda Saha, Kaushik Roy

**Abstract**: Label corruption, where training samples have incorrect labels, can significantly degrade the performance of machine learning models. This corruption often arises from non-expert labeling or adversarial attacks. Acquiring large, perfectly labeled datasets is costly, and retraining large models from scratch when a clean dataset becomes available is computationally expensive. To address this challenge, we propose Post-Training Correction, a new paradigm that adjusts model parameters after initial training to mitigate label noise, eliminating the need for retraining. We introduce Verifix, a novel Singular Value Decomposition (SVD) based algorithm that leverages a small, verified dataset to correct the model weights using a single update. Verifix uses SVD to estimate a Clean Activation Space and then projects the model's weights onto this space to suppress activations corresponding to corrupted data. We demonstrate Verifix's effectiveness on both synthetic and real-world label noise. Experiments on the CIFAR dataset with 25% synthetic corruption show 7.36% generalization improvements on average. Additionally, we observe generalization improvements of up to 2.63% on naturally corrupted datasets like WebVision1.0 and Clothing1M.



## **12. Continual Adversarial Defense**

cs.CV

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2312.09481v2) [paper-pdf](http://arxiv.org/pdf/2312.09481v2)

**Authors**: Qian Wang, Yaoyao Liu, Hefei Ling, Yingwei Li, Qihao Liu, Ping Li, Jiazhong Chen, Alan Yuille, Ning Yu

**Abstract**: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. The defense system must gather online few-shot defense feedback to promptly enhance itself, leveraging efficient memory utilization. Therefore, we propose the first continual adversarial defense (CAD) framework that adapts to any attacks in a dynamic scenario, where various attacks emerge stage by stage. In practice, CAD is modeled under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accuracy on both clean and adversarial images. We explore and integrate cutting-edge continual learning, few-shot learning, and ensemble learning techniques to qualify the principles. Experiments conducted on CIFAR-10 and ImageNet-100 validate the effectiveness of our approach against multiple stages of modern adversarial attacks and demonstrate significant improvements over numerous baseline methods. In particular, CAD is capable of quickly adapting with minimal feedback and a low cost of defense failure, while maintaining good performance against previous attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.



## **13. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

cs.CR

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2312.03853v2) [paper-pdf](http://arxiv.org/pdf/2312.03853v2)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Only a year ago, we witnessed a rise in the use of Large Language Models (LLMs), especially when combined with applications like chatbot assistants. Safety mechanisms and specialized training procedures are implemented to prevent improper responses from these assistants. In this work, we bypass these measures for ChatGPT and Bard (and, to some extent, Bing chat) by making them impersonate complex personas with opposite characteristics as those of the truthful assistants they are supposed to be. We start by creating elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversation followed a role-play style to get the response the assistant was not allowed to provide. By making use of personas, we show that the response that is prohibited is actually provided, making it possible to obtain unauthorized, illegal, or harmful information. This work shows that by using adversarial personas, one can overcome safety mechanisms set out by ChatGPT and Bard. We also introduce several ways of activating such adversarial personas, altogether showing that both chatbots are vulnerable to this kind of attack. With the same principle, we introduce two defenses that push the model to interpret trustworthy personalities and make it more robust against such attacks.



## **14. An Extended View on Measuring Tor AS-level Adversaries**

cs.NI

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08517v1) [paper-pdf](http://arxiv.org/pdf/2403.08517v1)

**Authors**: Gabriel Karl Gegenhuber, Markus Maier, Florian Holzbauer, Wilfried Mayer, Georg Merzdovnik, Edgar Weippl, Johanna Ullrich

**Abstract**: Tor provides anonymity to millions of users around the globe which has made it a valuable target for malicious actors. As a low-latency anonymity system, it is vulnerable to traffic correlation attacks from strong passive adversaries such as large autonomous systems (ASes). In preliminary work, we have developed a measurement approach utilizing the RIPE Atlas framework -- a network of more than 11,000 probes worldwide -- to infer the risk of deanonymization for IPv4 clients in Germany and the US.   In this paper, we apply our methodology to additional scenarios providing a broader picture of the potential for deanonymization in the Tor network. In particular, we (a) repeat our earlier (2020) measurements in 2022 to observe changes over time, (b) adopt our approach for IPv6 to analyze the risk of deanonymization when using this next-generation Internet protocol, and (c) investigate the current situation in Russia, where censorship has been intensified after the beginning of Russia's full-scale invasion of Ukraine. According to our results, Tor provides user anonymity at consistent quality: While individual numbers vary in dependence of client and destination, we were able to identify ASes with the potential to conduct deanonymization attacks. For clients in Germany and the US, the overall picture, however, has not changed since 2020. In addition, the protocols (IPv4 vs. IPv6) do not significantly impact the risk of deanonymization. Russian users are able to securely evade censorship using Tor. Their general risk of deanonymization is, in fact, lower than in the other investigated countries. Beyond, the few ASes with the potential to successfully perform deanonymization are operated by Western companies, further reducing the risk for Russian users.



## **15. The Philosopher's Stone: Trojaning Plugins of Large Language Models**

cs.CR

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2312.00374v2) [paper-pdf](http://arxiv.org/pdf/2312.00374v2)

**Authors**: Tian Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Shaofeng Li, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. In our experiments, we first conduct two case studies to demonstrate that a compromised LLM agent can execute malware to control system (e.g., LLM-driven robot) or launch a spear-phishing attack. Then, in terms of targeted misinformation, we show that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserve or improve the adapter's utility. Finally, we design and evaluate three potential defenses, yet none proved entirely effective in safeguarding against our attacks.



## **16. Fast Inference of Removal-Based Node Influence**

cs.LG

To be published in the Web Conference 2024

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08333v1) [paper-pdf](http://arxiv.org/pdf/2403.08333v1)

**Authors**: Weikai Li, Zhiping Xiao, Xiao Luo, Yizhou Sun

**Abstract**: Graph neural networks (GNNs) are widely utilized to capture the information spreading patterns in graphs. While remarkable performance has been achieved, there is a new trending topic of evaluating node influence. We propose a new method of evaluating node influence, which measures the prediction change of a trained GNN model caused by removing a node. A real-world application is, "In the task of predicting Twitter accounts' polarity, had a particular account been removed, how would others' polarity change?". We use the GNN as a surrogate model whose prediction could simulate the change of nodes or edges caused by node removal. To obtain the influence for every node, a straightforward way is to alternately remove every node and apply the trained GNN on the modified graph. It is reliable but time-consuming, so we need an efficient method. The related lines of work, such as graph adversarial attack and counterfactual explanation, cannot directly satisfy our needs, since they do not focus on the global influence score for every node. We propose an efficient and intuitive method, NOde-Removal-based fAst GNN inference (NORA), which uses the gradient to approximate the node-removal influence. It only costs one forward propagation and one backpropagation to approximate the influence score for all nodes. Extensive experiments on six datasets and six GNN models verify the effectiveness of NORA. Our code is available at https://github.com/weikai-li/NORA.git.



## **17. TSFool: Crafting Highly-Imperceptible Adversarial Time Series through Multi-Objective Attack**

cs.LG

22 pages, 16 figures

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2209.06388v3) [paper-pdf](http://arxiv.org/pdf/2209.06388v3)

**Authors**: Yanyun Wang, Dehui Du, Haibo Hu, Zi Liang, Yuanhao Liu

**Abstract**: Recent years have witnessed the success of recurrent neural network (RNN) models in time series classification (TSC). However, neural networks (NNs) are vulnerable to adversarial samples, which cause real-life adversarial attacks that undermine the robustness of AI models. To date, most existing attacks target at feed-forward NNs and image recognition tasks, but they cannot perform well on RNN-based TSC. This is due to the cyclical computation of RNN, which prevents direct model differentiation. In addition, the high visual sensitivity of time series to perturbations also poses challenges to local objective optimization of adversarial samples. In this paper, we propose an efficient method called TSFool to craft highly-imperceptible adversarial time series for RNN-based TSC. The core idea is a new global optimization objective known as "Camouflage Coefficient" that captures the imperceptibility of adversarial samples from the class distribution. Based on this, we reduce the adversarial attack problem to a multi-objective optimization problem that enhances the perturbation quality. Furthermore, to speed up the optimization process, we propose to use a representation model for RNN to capture deeply embedded vulnerable samples whose features deviate from the latent manifold. Experiments on 11 UCR and UEA datasets showcase that TSFool significantly outperforms six white-box and three black-box benchmark attacks in terms of effectiveness, efficiency and imperceptibility from various perspectives including standard measure, human study and real-world defense.



## **18. Attack Deterministic Conditional Image Generative Models for Diverse and Controllable Generation**

cs.CV

9 pages, 7 figures, accepted by AAAI24

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08294v1) [paper-pdf](http://arxiv.org/pdf/2403.08294v1)

**Authors**: Tianyi Chu, Wei Xing, Jiafu Chen, Zhizhong Wang, Jiakai Sun, Lei Zhao, Haibo Chen, Huaizhong Lin

**Abstract**: Existing generative adversarial network (GAN) based conditional image generative models typically produce fixed output for the same conditional input, which is unreasonable for highly subjective tasks, such as large-mask image inpainting or style transfer. On the other hand, GAN-based diverse image generative methods require retraining/fine-tuning the network or designing complex noise injection functions, which is computationally expensive, task-specific, or struggle to generate high-quality results. Given that many deterministic conditional image generative models have been able to produce high-quality yet fixed results, we raise an intriguing question: is it possible for pre-trained deterministic conditional image generative models to generate diverse results without changing network structures or parameters? To answer this question, we re-examine the conditional image generation tasks from the perspective of adversarial attack and propose a simple and efficient plug-in projected gradient descent (PGD) like method for diverse and controllable image generation. The key idea is attacking the pre-trained deterministic generative models by adding a micro perturbation to the input condition. In this way, diverse results can be generated without any adjustment of network structures or fine-tuning of the pre-trained models. In addition, we can also control the diverse results to be generated by specifying the attack direction according to a reference text or image. Our work opens the door to applying adversarial attack to low-level vision tasks, and experiments on various conditional image generation tasks demonstrate the effectiveness and superiority of the proposed method.



## **19. Versatile Defense Against Adversarial Attacks on Image Recognition**

cs.CV

**SubmitDate**: 2024-03-13    [abs](http://arxiv.org/abs/2403.08170v1) [paper-pdf](http://arxiv.org/pdf/2403.08170v1)

**Authors**: Haibo Zhang, Zhihua Yao, Kouichi Sakurai

**Abstract**: Adversarial attacks present a significant security risk to image recognition tasks. Defending against these attacks in a real-life setting can be compared to the way antivirus software works, with a key consideration being how well the defense can adapt to new and evolving attacks. Another important factor is the resources involved in terms of time and cost for training defense models and updating the model database. Training many models that are specific to each type of attack can be time-consuming and expensive. Ideally, we should be able to train one single model that can handle a wide range of attacks. It appears that a defense method based on image-to-image translation may be capable of this. The proposed versatile defense approach in this paper only requires training one model to effectively resist various unknown adversarial attacks. The trained model has successfully improved the classification accuracy from nearly zero to an average of 86%, performing better than other defense methods proposed in prior studies. When facing the PGD attack and the MI-FGSM attack, versatile defense model even outperforms the attack-specific models trained based on these two attacks. The robustness check also shows that our versatile defense model performs stably regardless with the attack strength.



## **20. Information Leakage through Physical Layer Supply Voltage Coupling Vulnerability**

cs.CR

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.08132v1) [paper-pdf](http://arxiv.org/pdf/2403.08132v1)

**Authors**: Sahan Sanjaya, Aruna Jayasena, Prabhat Mishra

**Abstract**: Side-channel attacks exploit variations in non-functional behaviors to expose sensitive information across security boundaries. Existing methods leverage side-channels based on power consumption, electromagnetic radiation, silicon substrate coupling, and channels created by malicious implants. Power-based side-channel attacks are widely known for extracting information from data processed within a device while assuming that an attacker has physical access or the ability to modify the device. In this paper, we introduce a novel side-channel vulnerability that leaks data-dependent power variations through physical layer supply voltage coupling (PSVC). Unlike traditional power side-channel attacks, the proposed vulnerability allows an adversary to mount an attack and extract information without modifying the device. We assess the effectiveness of PSVC vulnerability through three case studies, demonstrating several end-to-end attacks on general-purpose microcontrollers with varying adversary capabilities. These case studies provide evidence for the existence of PSVC vulnerability, its applicability for on-chip as well as on-board side-channel attacks, and how it can eliminate the need for physical access to the target device, making it applicable to any off-the-shelf hardware. Our experiments also reveal that designing devices to operate at the lowest operational voltage significantly reduces the risk of PSVC side-channel vulnerability.



## **21. Quantifying and Mitigating Privacy Risks for Tabular Generative Models**

cs.LG

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.07842v1) [paper-pdf](http://arxiv.org/pdf/2403.07842v1)

**Authors**: Chaoyi Zhu, Jiayi Tang, Hans Brouwer, Juan F. Pérez, Marten van Dijk, Lydia Y. Chen

**Abstract**: Synthetic data from generative models emerges as the privacy-preserving data-sharing solution. Such a synthetic data set shall resemble the original data without revealing identifiable private information. The backbone technology of tabular synthesizers is rooted in image generative models, ranging from Generative Adversarial Networks (GANs) to recent diffusion models. Recent prior work sheds light on the utility-privacy tradeoff on tabular data, revealing and quantifying privacy risks on synthetic data. We first conduct an exhaustive empirical analysis, highlighting the utility-privacy tradeoff of five state-of-the-art tabular synthesizers, against eight privacy attacks, with a special focus on membership inference attacks. Motivated by the observation of high data quality but also high privacy risk in tabular diffusion, we propose DP-TLDM, Differentially Private Tabular Latent Diffusion Model, which is composed of an autoencoder network to encode the tabular data and a latent diffusion model to synthesize the latent tables. Following the emerging f-DP framework, we apply DP-SGD to train the auto-encoder in combination with batch clipping and use the separation value as the privacy metric to better capture the privacy gain from DP algorithms. Our empirical evaluation demonstrates that DP-TLDM is capable of achieving a meaningful theoretical privacy guarantee while also significantly enhancing the utility of synthetic data. Specifically, compared to other DP-protected tabular generative models, DP-TLDM improves the synthetic quality by an average of 35% in data resemblance, 15% in the utility for downstream tasks, and 50% in data discriminability, all while preserving a comparable level of privacy risk.



## **22. Robustifying Point Cloud Networks by Refocusing**

cs.CV

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2308.05525v3) [paper-pdf](http://arxiv.org/pdf/2308.05525v3)

**Authors**: Meir Yossef Levi, Guy Gilboa

**Abstract**: The ability to cope with out-of-distribution (OOD) corruptions and adversarial attacks is crucial in real-world safety-demanding applications. In this study, we develop a general mechanism to increase neural network robustness based on focus analysis.   Recent studies have revealed the phenomenon of \textit{Overfocusing}, which leads to a performance drop. When the network is primarily influenced by small input regions, it becomes less robust and prone to misclassify under noise and corruptions.   However, quantifying overfocusing is still vague and lacks clear definitions. Here, we provide a mathematical definition of \textbf{focus}, \textbf{overfocusing} and \textbf{underfocusing}. The notions are general, but in this study, we specifically investigate the case of 3D point clouds.   We observe that corrupted sets result in a biased focus distribution compared to the clean training set.   We show that as focus distribution deviates from the one learned in the training phase - classification performance deteriorates.   We thus propose a parameter-free \textbf{refocusing} algorithm that aims to unify all corruptions under the same distribution.   We validate our findings on a 3D zero-shot classification task, achieving SOTA in robust 3D classification on ModelNet-C dataset, and in adversarial defense against Shape-Invariant attack. Code is available in: https://github.com/yossilevii100/refocusing.



## **23. Analyzing Adversarial Attacks on Sequence-to-Sequence Relevance Models**

cs.IR

13 pages, 3 figures, Accepted at ECIR 2024 as a Full Paper

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.07654v1) [paper-pdf](http://arxiv.org/pdf/2403.07654v1)

**Authors**: Andrew Parry, Maik Fröbe, Sean MacAvaney, Martin Potthast, Matthias Hagen

**Abstract**: Modern sequence-to-sequence relevance models like monoT5 can effectively capture complex textual interactions between queries and documents through cross-encoding. However, the use of natural language tokens in prompts, such as Query, Document, and Relevant for monoT5, opens an attack vector for malicious documents to manipulate their relevance score through prompt injection, e.g., by adding target words such as true. Since such possibilities have not yet been considered in retrieval evaluation, we analyze the impact of query-independent prompt injection via manually constructed templates and LLM-based rewriting of documents on several existing relevance models. Our experiments on the TREC Deep Learning track show that adversarial documents can easily manipulate different sequence-to-sequence relevance models, while BM25 (as a typical lexical model) is not affected. Remarkably, the attacks also affect encoder-only relevance models (which do not rely on natural language prompt tokens), albeit to a lesser extent.



## **24. Visual Privacy Auditing with Diffusion Models**

cs.LG

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.07588v1) [paper-pdf](http://arxiv.org/pdf/2403.07588v1)

**Authors**: Kristian Schwethelm, Johannes Kaiser, Moritz Knolle, Daniel Rueckert, Georgios Kaissis, Alexander Ziller

**Abstract**: Image reconstruction attacks on machine learning models pose a significant risk to privacy by potentially leaking sensitive information. Although defending against such attacks using differential privacy (DP) has proven effective, determining appropriate DP parameters remains challenging. Current formal guarantees on data reconstruction success suffer from overly theoretical assumptions regarding adversary knowledge about the target data, particularly in the image domain. In this work, we empirically investigate this discrepancy and find that the practicality of these assumptions strongly depends on the domain shift between the data prior and the reconstruction target. We propose a reconstruction attack based on diffusion models (DMs) that assumes adversary access to real-world image priors and assess its implications on privacy leakage under DP-SGD. We show that (1) real-world data priors significantly influence reconstruction success, (2) current reconstruction bounds do not model the risk posed by data priors well, and (3) DMs can serve as effective auditing tools for visualizing privacy leakage.



## **25. Improving deep learning with prior knowledge and cognitive models: A survey on enhancing explainability, adversarial robustness and zero-shot learning**

cs.LG

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.07078v1) [paper-pdf](http://arxiv.org/pdf/2403.07078v1)

**Authors**: Fuseinin Mumuni, Alhassan Mumuni

**Abstract**: We review current and emerging knowledge-informed and brain-inspired cognitive systems for realizing adversarial defenses, eXplainable Artificial Intelligence (XAI), and zero-shot or few-short learning. Data-driven deep learning models have achieved remarkable performance and demonstrated capabilities surpassing human experts in many applications. Yet, their inability to exploit domain knowledge leads to serious performance limitations in practical applications. In particular, deep learning systems are exposed to adversarial attacks, which can trick them into making glaringly incorrect decisions. Moreover, complex data-driven models typically lack interpretability or explainability, i.e., their decisions cannot be understood by human subjects. Furthermore, models are usually trained on standard datasets with a closed-world assumption. Hence, they struggle to generalize to unseen cases during inference in practical open-world environments, thus, raising the zero- or few-shot generalization problem. Although many conventional solutions exist, explicit domain knowledge, brain-inspired neural network and cognitive architectures offer powerful new dimensions towards alleviating these problems. Prior knowledge is represented in appropriate forms and incorporated in deep learning frameworks to improve performance. Brain-inspired cognition methods use computational models that mimic the human mind to enhance intelligent behavior in artificial agents and autonomous robots. Ultimately, these models achieve better explainability, higher adversarial robustness and data-efficient learning, and can, in turn, provide insights for cognitive science and neuroscience-that is, to deepen human understanding on how the brain works in general, and how it handles these problems.



## **26. Enhancing Adversarial Training with Prior Knowledge Distillation for Robust Image Compression**

eess.IV

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06700v1) [paper-pdf](http://arxiv.org/pdf/2403.06700v1)

**Authors**: Cao Zhi, Bao Youneng, Meng Fanyang, Li Chao, Tan Wen, Wang Genhong, Liang Yongsheng

**Abstract**: Deep neural network-based image compression (NIC) has achieved excellent performance, but NIC method models have been shown to be susceptible to backdoor attacks. Adversarial training has been validated in image compression models as a common method to enhance model robustness. However, the improvement effect of adversarial training on model robustness is limited. In this paper, we propose a prior knowledge-guided adversarial training framework for image compression models. Specifically, first, we propose a gradient regularization constraint for training robust teacher models. Subsequently, we design a knowledge distillation based strategy to generate a priori knowledge from the teacher model to the student model for guiding adversarial training. Experimental results show that our method improves the reconstruction quality by about 9dB when the Kodak dataset is elected as the backdoor attack object for psnr attack. Compared with Ma2023, our method has a 5dB higher PSNR output at high bitrate points.



## **27. PCLD: Point Cloud Layerwise Diffusion for Adversarial Purification**

cs.CV

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06698v1) [paper-pdf](http://arxiv.org/pdf/2403.06698v1)

**Authors**: Mert Gulsen, Batuhan Cengiz, Yusuf H. Sahin, Gozde Unal

**Abstract**: Point clouds are extensively employed in a variety of real-world applications such as robotics, autonomous driving and augmented reality. Despite the recent success of point cloud neural networks, especially for safety-critical tasks, it is essential to also ensure the robustness of the model. A typical way to assess a model's robustness is through adversarial attacks, where test-time examples are generated based on gradients to deceive the model. While many different defense mechanisms are studied in 2D, studies on 3D point clouds have been relatively limited in the academic field. Inspired from PointDP, which denoises the network inputs by diffusion, we propose Point Cloud Layerwise Diffusion (PCLD), a layerwise diffusion based 3D point cloud defense strategy. Unlike PointDP, we propagated the diffusion denoising after each layer to incrementally enhance the results. We apply our defense method to different types of commonly used point cloud models and adversarial attacks to evaluate its robustness. Our experiments demonstrate that the proposed defense method achieved results that are comparable to or surpass those of existing methodologies, establishing robustness through a novel technique. Code is available at https://github.com/batuceng/diffusion-layer-robustness-pc.



## **28. PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor**

cs.LG

Accepted to CVPR 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06668v1) [paper-pdf](http://arxiv.org/pdf/2403.06668v1)

**Authors**: Jaewon Jung, Hongsun Jang, Jaeyong Song, Jinho Lee

**Abstract**: Adversarial robustness of the neural network is a significant concern when it is applied to security-critical domains. In this situation, adversarial distillation is a promising option which aims to distill the robustness of the teacher network to improve the robustness of a small student network. Previous works pretrain the teacher network to make it robust to the adversarial examples aimed at itself. However, the adversarial examples are dependent on the parameters of the target network. The fixed teacher network inevitably degrades its robustness against the unseen transferred adversarial examples which targets the parameters of the student network in the adversarial distillation process. We propose PeerAiD to make a peer network learn the adversarial examples of the student network instead of adversarial examples aimed at itself. PeerAiD is an adversarial distillation that trains the peer network and the student network simultaneously in order to make the peer network specialized for defending the student network. We observe that such peer networks surpass the robustness of pretrained robust teacher network against student-attacked adversarial samples. With this peer network and adversarial distillation, PeerAiD achieves significantly higher robustness of the student network with AutoAttack (AA) accuracy up to 1.66%p and improves the natural accuracy of the student network up to 4.72%p with ResNet-18 and TinyImageNet dataset.



## **29. epsilon-Mesh Attack: A Surface-based Adversarial Point Cloud Attack for Facial Expression Recognition**

cs.CV

Accepted at 18th IEEE International Conference on Automatic Face &  Gesture Recognition (FG 2024)

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06661v1) [paper-pdf](http://arxiv.org/pdf/2403.06661v1)

**Authors**: Batuhan Cengiz, Mert Gulsen, Yusuf H. Sahin, Gozde Unal

**Abstract**: Point clouds and meshes are widely used 3D data structures for many computer vision applications. While the meshes represent the surfaces of an object, point cloud represents sampled points from the surface which is also the output of modern sensors such as LiDAR and RGB-D cameras. Due to the wide application area of point clouds and the recent advancements in deep neural networks, studies focusing on robust classification of the 3D point cloud data emerged. To evaluate the robustness of deep classifier networks, a common method is to use adversarial attacks where the gradient direction is followed to change the input slightly. The previous studies on adversarial attacks are generally evaluated on point clouds of daily objects. However, considering 3D faces, these adversarial attacks tend to affect the person's facial structure more than the desired amount and cause malformation. Specifically for facial expressions, even a small adversarial attack can have a significant effect on the face structure. In this paper, we suggest an adversarial attack called $\epsilon$-Mesh Attack, which operates on point cloud data via limiting perturbations to be on the mesh surface. We also parameterize our attack by $\epsilon$ to scale the perturbation mesh. Our surface-based attack has tighter perturbation bounds compared to $L_2$ and $L_\infty$ norm bounded attacks that operate on unit-ball. Even though our method has additional constraints, our experiments on CoMA, Bosphorus and FaceWarehouse datasets show that $\epsilon$-Mesh Attack (Perpendicular) successfully confuses trained DGCNN and PointNet models $99.72\%$ and $97.06\%$ of the time, with indistinguishable facial deformations. The code is available at https://github.com/batuceng/e-mesh-attack.



## **30. Real is not True: Backdoor Attacks Against Deepfake Detection**

cs.CR

BigDIA 2023

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06610v1) [paper-pdf](http://arxiv.org/pdf/2403.06610v1)

**Authors**: Hong Sun, Ziqiang Li, Lei Liu, Bin Li

**Abstract**: The proliferation of malicious deepfake applications has ignited substantial public apprehension, casting a shadow of doubt upon the integrity of digital media. Despite the development of proficient deepfake detection mechanisms, they persistently demonstrate pronounced vulnerability to an array of attacks. It is noteworthy that the pre-existing repertoire of attacks predominantly comprises adversarial example attack, predominantly manifesting during the testing phase. In the present study, we introduce a pioneering paradigm denominated as Bad-Deepfake, which represents a novel foray into the realm of backdoor attacks levied against deepfake detectors. Our approach hinges upon the strategic manipulation of a delimited subset of the training data, enabling us to wield disproportionate influence over the operational characteristics of a trained model. This manipulation leverages inherent frailties inherent to deepfake detectors, affording us the capacity to engineer triggers and judiciously select the most efficacious samples for the construction of the poisoned set. Through the synergistic amalgamation of these sophisticated techniques, we achieve an remarkable performance-a 100% attack success rate (ASR) against extensively employed deepfake detectors.



## **31. DNNShield: Embedding Identifiers for Deep Neural Network Ownership Verification**

cs.CR

18 pages, 11 figures, 6 tables

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06581v1) [paper-pdf](http://arxiv.org/pdf/2403.06581v1)

**Authors**: Jasper Stang, Torsten Krauß, Alexandra Dmitrienko

**Abstract**: The surge in popularity of machine learning (ML) has driven significant investments in training Deep Neural Networks (DNNs). However, these models that require resource-intensive training are vulnerable to theft and unauthorized use. This paper addresses this challenge by introducing DNNShield, a novel approach for DNN protection that integrates seamlessly before training. DNNShield embeds unique identifiers within the model architecture using specialized protection layers. These layers enable secure training and deployment while offering high resilience against various attacks, including fine-tuning, pruning, and adaptive adversarial attacks. Notably, our approach achieves this security with minimal performance and computational overhead (less than 5\% runtime increase). We validate the effectiveness and efficiency of DNNShield through extensive evaluations across three datasets and four model architectures. This practical solution empowers developers to protect their DNNs and intellectual property rights.



## **32. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

cs.SD

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2305.17000v3) [paper-pdf](http://arxiv.org/pdf/2305.17000v3)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.



## **33. Fooling Neural Networks for Motion Forecasting via Adversarial Attacks**

cs.CV

11 pages, 8 figures, VISSAP 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.04954v2) [paper-pdf](http://arxiv.org/pdf/2403.04954v2)

**Authors**: Edgar Medina, Leyong Loh

**Abstract**: Human motion prediction is still an open problem, which is extremely important for autonomous driving and safety applications. Although there are great advances in this area, the widely studied topic of adversarial attacks has not been applied to multi-regression models such as GCNs and MLP-based architectures in human motion prediction. This work intends to reduce this gap using extensive quantitative and qualitative experiments in state-of-the-art architectures similar to the initial stages of adversarial attacks in image classification. The results suggest that models are susceptible to attacks even on low levels of perturbation. We also show experiments with 3D transformations that affect the model performance, in particular, we show that most models are sensitive to simple rotations and translations which do not alter joint distances. We conclude that similar to earlier CNN models, motion forecasting tasks are susceptible to small perturbations and simple 3D transformations.



## **34. Intra-Section Code Cave Injection for Adversarial Evasion Attacks on Windows PE Malware File**

cs.CR

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06428v1) [paper-pdf](http://arxiv.org/pdf/2403.06428v1)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam, Moustafa Saleh

**Abstract**: Windows malware is predominantly available in cyberspace and is a prime target for deliberate adversarial evasion attacks. Although researchers have investigated the adversarial malware attack problem, a multitude of important questions remain unanswered, including (a) Are the existing techniques to inject adversarial perturbations in Windows Portable Executable (PE) malware files effective enough for evasion purposes?; (b) Does the attack process preserve the original behavior of malware?; (c) Are there unexplored approaches/locations that can be used to carry out adversarial evasion attacks on Windows PE malware?; and (d) What are the optimal locations and sizes of adversarial perturbations required to evade an ML-based malware detector without significant structural change in the PE file? To answer some of these questions, this work proposes a novel approach that injects a code cave within the section (i.e., intra-section) of Windows PE malware files to make space for adversarial perturbations. In addition, a code loader is also injected inside the PE file, which reverts adversarial malware to its original form during the execution, preserving the malware's functionality and executability. To understand the effectiveness of our approach, we injected adversarial perturbations inside the .text, .data and .rdata sections, generated using the gradient descent and Fast Gradient Sign Method (FGSM), to target the two popular CNN-based malware detectors, MalConv and MalConv2. Our experiments yielded notable results, achieving a 92.31% evasion rate with gradient descent and 96.26% with FGSM against MalConv, compared to the 16.17% evasion rate for append attacks. Similarly, when targeting MalConv2, our approach achieved a remarkable maximum evasion rate of 97.93% with gradient descent and 94.34% with FGSM, significantly surpassing the 4.01% evasion rate observed with append attacks.



## **35. A Zero Trust Framework for Realization and Defense Against Generative AI Attacks in Power Grid**

cs.CR

Accepted article by IEEE International Conference on Communications  (ICC 2024), Copyright 2024 IEEE

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06388v1) [paper-pdf](http://arxiv.org/pdf/2403.06388v1)

**Authors**: Md. Shirajum Munir, Sravanthi Proddatoori, Manjushree Muralidhara, Walid Saad, Zhu Han, Sachin Shetty

**Abstract**: Understanding the potential of generative AI (GenAI)-based attacks on the power grid is a fundamental challenge that must be addressed in order to protect the power grid by realizing and validating risk in new attack vectors. In this paper, a novel zero trust framework for a power grid supply chain (PGSC) is proposed. This framework facilitates early detection of potential GenAI-driven attack vectors (e.g., replay and protocol-type attacks), assessment of tail risk-based stability measures, and mitigation of such threats. First, a new zero trust system model of PGSC is designed and formulated as a zero-trust problem that seeks to guarantee for a stable PGSC by realizing and defending against GenAI-driven cyber attacks. Second, in which a domain-specific generative adversarial networks (GAN)-based attack generation mechanism is developed to create a new vulnerability cyberspace for further understanding that threat. Third, tail-based risk realization metrics are developed and implemented for quantifying the extreme risk of a potential attack while leveraging a trust measurement approach for continuous validation. Fourth, an ensemble learning-based bootstrap aggregation scheme is devised to detect the attacks that are generating synthetic identities with convincing user and distributed energy resources device profiles. Experimental results show the efficacy of the proposed zero trust framework that achieves an accuracy of 95.7% on attack vector generation, a risk measure of 9.61% for a 95% stable PGSC, and a 99% confidence in defense against GenAI-driven attack.



## **36. Towards Scalable and Robust Model Versioning**

cs.LG

Published in IEEE SaTML 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2401.09574v2) [paper-pdf](http://arxiv.org/pdf/2401.09574v2)

**Authors**: Wenxin Ding, Arjun Nitin Bhagoji, Ben Y. Zhao, Haitao Zheng

**Abstract**: As the deployment of deep learning models continues to expand across industries, the threat of malicious incursions aimed at gaining access to these deployed models is on the rise. Should an attacker gain access to a deployed model, whether through server breaches, insider attacks, or model inversion techniques, they can then construct white-box adversarial attacks to manipulate the model's classification outcomes, thereby posing significant risks to organizations that rely on these models for critical tasks. Model owners need mechanisms to protect themselves against such losses without the necessity of acquiring fresh training data - a process that typically demands substantial investments in time and capital.   In this paper, we explore the feasibility of generating multiple versions of a model that possess different attack properties, without acquiring new training data or changing model architecture. The model owner can deploy one version at a time and replace a leaked version immediately with a new version. The newly deployed model version can resist adversarial attacks generated leveraging white-box access to one or all previously leaked versions. We show theoretically that this can be accomplished by incorporating parameterized hidden distributions into the model training data, forcing the model to learn task-irrelevant features uniquely defined by the chosen data. Additionally, optimal choices of hidden distributions can produce a sequence of model versions capable of resisting compound transferability attacks over time. Leveraging our analytical insights, we design and implement a practical model versioning method for DNN classifiers, which leads to significant robustness improvements over existing methods. We believe our work presents a promising direction for safeguarding DNN services beyond their initial deployment.



## **37. Fake or Compromised? Making Sense of Malicious Clients in Federated Learning**

cs.LG

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2403.06319v1) [paper-pdf](http://arxiv.org/pdf/2403.06319v1)

**Authors**: Hamid Mozaffari, Sunav Choudhary, Amir Houmansadr

**Abstract**: Federated learning (FL) is a distributed machine learning paradigm that enables training models on decentralized data. The field of FL security against poisoning attacks is plagued with confusion due to the proliferation of research that makes different assumptions about the capabilities of adversaries and the adversary models they operate under. Our work aims to clarify this confusion by presenting a comprehensive analysis of the various poisoning attacks and defensive aggregation rules (AGRs) proposed in the literature, and connecting them under a common framework. To connect existing adversary models, we present a hybrid adversary model, which lies in the middle of the spectrum of adversaries, where the adversary compromises a few clients, trains a generative (e.g., DDPM) model with their compromised samples, and generates new synthetic data to solve an optimization for a stronger (e.g., cheaper, more practical) attack against different robust aggregation rules. By presenting the spectrum of FL adversaries, we aim to provide practitioners and researchers with a clear understanding of the different types of threats they need to consider when designing FL systems, and identify areas where further research is needed.



## **38. Improving behavior based authentication against adversarial attack using XAI**

cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2402.16430v2) [paper-pdf](http://arxiv.org/pdf/2402.16430v2)

**Authors**: Dong Qin, George Amariucai, Daji Qiao, Yong Guan

**Abstract**: In recent years, machine learning models, especially deep neural networks, have been widely used for classification tasks in the security domain. However, these models have been shown to be vulnerable to adversarial manipulation: small changes learned by an adversarial attack model, when applied to the input, can cause significant changes in the output. Most research on adversarial attacks and corresponding defense methods focuses only on scenarios where adversarial samples are directly generated by the attack model. In this study, we explore a more practical scenario in behavior-based authentication, where adversarial samples are collected from the attacker. The generated adversarial samples from the model are replicated by attackers with a certain level of discrepancy. We propose an eXplainable AI (XAI) based defense strategy against adversarial attacks in such scenarios. A feature selector, trained with our method, can be used as a filter in front of the original authenticator. It filters out features that are more vulnerable to adversarial attacks or irrelevant to authentication, while retaining features that are more robust. Through comprehensive experiments, we demonstrate that our XAI based defense strategy is effective against adversarial attacks and outperforms other defense strategies, such as adversarial training and defensive distillation.



## **39. Learn from the Past: A Proxy Guided Adversarial Defense Framework with Self Distillation Regularization**

cs.LG

13 Pages

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.12713v2) [paper-pdf](http://arxiv.org/pdf/2310.12713v2)

**Authors**: Yaohua Liu, Jiaxin Gao, Xianghao Jiao, Zhu Liu, Xin Fan, Risheng Liu

**Abstract**: Adversarial Training (AT), pivotal in fortifying the robustness of deep learning models, is extensively adopted in practical applications. However, prevailing AT methods, relying on direct iterative updates for target model's defense, frequently encounter obstacles such as unstable training and catastrophic overfitting. In this context, our work illuminates the potential of leveraging the target model's historical states as a proxy to provide effective initialization and defense prior, which results in a general proxy guided defense framework, `LAST' ({\bf L}earn from the P{\bf ast}). Specifically, LAST derives response of the proxy model as dynamically learned fast weights, which continuously corrects the update direction of the target model. Besides, we introduce a self-distillation regularized defense objective, ingeniously designed to steer the proxy model's update trajectory without resorting to external teacher models, thereby ameliorating the impact of catastrophic overfitting on performance. Extensive experiments and ablation studies showcase the framework's efficacy in markedly improving model robustness (e.g., up to 9.2\% and 20.3\% enhancement in robust accuracy on CIFAR10 and CIFAR100 datasets, respectively) and training stability. These improvements are consistently observed across various model architectures, larger datasets, perturbation sizes, and attack modalities, affirming LAST's ability to consistently refine both single-step and multi-step AT strategies. The code will be available at~\url{https://github.com/callous-youth/LAST}.



## **40. Deep Reinforcement Learning with Spiking Q-learning**

cs.NE

15 pages, 7 figures

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2201.09754v2) [paper-pdf](http://arxiv.org/pdf/2201.09754v2)

**Authors**: Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian

**Abstract**: With the help of special neuromorphic hardware, spiking neural networks (SNNs) are expected to realize artificial intelligence (AI) with less energy consumption. It provides a promising energy-efficient way for realistic control tasks by combining SNNs with deep reinforcement learning (RL). There are only a few existing SNN-based RL methods at present. Most of them either lack generalization ability or employ Artificial Neural Networks (ANNs) to estimate value function in training. The former needs to tune numerous hyper-parameters for each scenario, and the latter limits the application of different types of RL algorithm and ignores the large energy consumption in training. To develop a robust spike-based RL method, we draw inspiration from non-spiking interneurons found in insects and propose the deep spiking Q-network (DSQN), using the membrane voltage of non-spiking neurons as the representation of Q-value, which can directly learn robust policies from high-dimensional sensory inputs using end-to-end RL. Experiments conducted on 17 Atari games demonstrate the DSQN is effective and even outperforms the ANN-based deep Q-network (DQN) in most games. Moreover, the experiments show superior learning stability and robustness to adversarial attacks of DSQN.



## **41. Language-Driven Anchors for Zero-Shot Adversarial Robustness**

cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2301.13096v3) [paper-pdf](http://arxiv.org/pdf/2301.13096v3)

**Authors**: Xiao Li, Wei Zhang, Yining Liu, Zhanhao Hu, Bo Zhang, Xiaolin Hu

**Abstract**: Deep Neural Networks (DNNs) are known to be susceptible to adversarial attacks. Previous researches mainly focus on improving adversarial robustness in the fully supervised setting, leaving the challenging domain of zero-shot adversarial robustness an open question. In this work, we investigate this domain by leveraging the recent advances in large vision-language models, such as CLIP, to introduce zero-shot adversarial robustness to DNNs. We propose LAAT, a Language-driven, Anchor-based Adversarial Training strategy. LAAT utilizes the features of a text encoder for each category as fixed anchors (normalized feature embeddings) for each category, which are then employed for adversarial training. By leveraging the semantic consistency of the text encoders, LAAT aims to enhance the adversarial robustness of the image model on novel categories. However, naively using text encoders leads to poor results. Through analysis, we identified the issue to be the high cosine similarity between text encoders. We then design an expansion algorithm and an alignment cross-entropy loss to alleviate the problem. Our experimental results demonstrated that LAAT significantly improves zero-shot adversarial robustness over state-of-the-art methods. LAAT has the potential to enhance adversarial robustness by large-scale multimodal models, especially when labeled data is unavailable during training.



## **42. Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**

cs.CV

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2401.16352v2) [paper-pdf](http://arxiv.org/pdf/2401.16352v2)

**Authors**: Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline called Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks resulting in the robustness generalization to unseen attacks and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves state-of-the-art results and exhibits generalization ability against unseen attacks.



## **43. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.19181v2) [paper-pdf](http://arxiv.org/pdf/2310.19181v2)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs, i.e., ChatGPT (GPT 3.5 Turbo), GPT 4, Claude, and Bard, to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing websites and emails that can convincingly imitate well-known brands and also deploy a range of evasive tactics that are used to elude detection mechanisms employed by anti-phishing systems. These attacks can be generated using unmodified or "vanilla" versions of these LLMs without requiring any prior adversarial exploits such as jailbreaking. We evaluate the performance of the LLMs towards generating these attacks and find that they can also be utilized to create malicious prompts that, in turn, can be fed back to the model to generate phishing scams - thus massively reducing the prompt-engineering effort required by attackers to scale these threats. As a countermeasure, we build a BERT-based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content. Our model is transferable across all four commercial LLMs, attaining an average accuracy of 96% for phishing website prompts and 94% for phishing email prompts. We also disclose the vulnerabilities to the concerned LLMs, with Google acknowledging it as a severe issue. Our detection model is available for use at Hugging Face, as well as a ChatGPT Actions plugin.



## **44. Attacking Transformers with Feature Diversity Adversarial Perturbation**

cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2403.07942v1) [paper-pdf](http://arxiv.org/pdf/2403.07942v1)

**Authors**: Chenxing Gao, Hang Zhou, Junqing Yu, YuTeng Ye, Jiale Cai, Junle Wang, Wei Yang

**Abstract**: Understanding the mechanisms behind Vision Transformer (ViT), particularly its vulnerability to adversarial perturba tions, is crucial for addressing challenges in its real-world applications. Existing ViT adversarial attackers rely on la bels to calculate the gradient for perturbation, and exhibit low transferability to other structures and tasks. In this paper, we present a label-free white-box attack approach for ViT-based models that exhibits strong transferability to various black box models, including most ViT variants, CNNs, and MLPs, even for models developed for other modalities. Our inspira tion comes from the feature collapse phenomenon in ViTs, where the critical attention mechanism overly depends on the low-frequency component of features, causing the features in middle-to-end layers to become increasingly similar and eventually collapse. We propose the feature diversity attacker to naturally accelerate this process and achieve remarkable performance and transferability.



## **45. Hard-label based Small Query Black-box Adversarial Attack**

cs.LG

11 pages, 3 figures

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2403.06014v1) [paper-pdf](http://arxiv.org/pdf/2403.06014v1)

**Authors**: Jeonghwan Park, Paul Miller, Niall McLaughlin

**Abstract**: We consider the hard label based black box adversarial attack setting which solely observes predicted classes from the target model. Most of the attack methods in this setting suffer from impractical number of queries required to achieve a successful attack. One approach to tackle this drawback is utilising the adversarial transferability between white box surrogate models and black box target model. However, the majority of the methods adopting this approach are soft label based to take the full advantage of zeroth order optimisation. Unlike mainstream methods, we propose a new practical setting of hard label based attack with an optimisation process guided by a pretrained surrogate model. Experiments show the proposed method significantly improves the query efficiency of the hard label based black-box attack across various target model architectures. We find the proposed method achieves approximately 5 times higher attack success rate compared to the benchmarks, especially at the small query budgets as 100 and 250.



## **46. IOI: Invisible One-Iteration Adversarial Attack on No-Reference Image- and Video-Quality Metrics**

eess.IV

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2403.05955v1) [paper-pdf](http://arxiv.org/pdf/2403.05955v1)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: No-reference image- and video-quality metrics are widely used in video processing benchmarks. The robustness of learning-based metrics under video attacks has not been widely studied. In addition to having success, attacks that can be employed in video processing benchmarks must be fast and imperceptible. This paper introduces an Invisible One-Iteration (IOI) adversarial attack on no reference image and video quality metrics. We compared our method alongside eight prior approaches using image and video datasets via objective and subjective tests. Our method exhibited superior visual quality across various attacked metric architectures while maintaining comparable attack success and speed. We made the code available on GitHub.



## **47. SoK: Secure Human-centered Wireless Sensing**

cs.CR

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2211.12087v2) [paper-pdf](http://arxiv.org/pdf/2211.12087v2)

**Authors**: Wei Sun, Tingjun Chen, Neil Gong

**Abstract**: Human-centered wireless sensing (HCWS) aims to understand the fine-grained environment and activities of a human using the diverse wireless signals around him/her. While the sensed information about a human can be used for many good purposes such as enhancing life quality, an adversary can also abuse it to steal private information about the human (e.g., location and person's identity). However, the literature lacks a systematic understanding of the privacy vulnerabilities of wireless sensing and the defenses against them, resulting in the privacy-compromising HCWS design.   In this work, we aim to bridge this gap to achieve the vision of secure human-centered wireless sensing. First, we propose a signal processing pipeline to identify private information leakage and further understand the benefits and tradeoffs of wireless sensing-based inference attacks and defenses. Based on this framework, we present the taxonomy of existing inference attacks and defenses. As a result, we can identify the open challenges and gaps in achieving privacy-preserving human-centered wireless sensing in the era of machine learning and further propose directions for future research in this field.



## **48. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

cs.RO

8 pages (7 content, 1 reference). 5 figures, submitted to the IEEE  Robotics and Automation Letters (RA-L)

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05666v1) [paper-pdf](http://arxiv.org/pdf/2403.05666v1)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method to assess the resilience of the Iterative Closest Point (ICP) algorithm via deep-learning-based attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms prior to deployments is of utmost importance. The ICP algorithm has become the standard for lidar-based localization. However, the pose estimate it produces can be greatly affected by corruption in the measurements. Corruption can arise from a variety of scenarios such as occlusions, adverse weather, or mechanical issues in the sensor. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP empirically, our method focuses on finding the maximum possible ICP pose error using perturbation-based adversarial attacks. The proposed attack induces significant pose errors on ICP and outperforms baselines more than 88% of the time across a wide range of scenarios. As an example application, we demonstrate that our attack can be used to identify areas on a map where ICP is particularly vulnerable to corruption in the measurements.



## **49. Invariant Aggregator for Defending against Federated Backdoor Attacks**

cs.LG

AISTATS 2024 camera-ready

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2210.01834v4) [paper-pdf](http://arxiv.org/pdf/2210.01834v4)

**Authors**: Xiaoyang Wang, Dimitrios Dimitriadis, Sanmi Koyejo, Shruti Tople

**Abstract**: Federated learning enables training high-utility models across several clients without directly sharing their private data. As a downside, the federated setting makes the model vulnerable to various adversarial attacks in the presence of malicious clients. Despite the theoretical and empirical success in defending against attacks that aim to degrade models' utility, defense against backdoor attacks that increase model accuracy on backdoor samples exclusively without hurting the utility on other samples remains challenging. To this end, we first analyze the failure modes of existing defenses over a flat loss landscape, which is common for well-designed neural networks such as Resnet (He et al., 2015) but is often overlooked by previous works. Then, we propose an invariant aggregator that redirects the aggregated update to invariant directions that are generally useful via selectively masking out the update elements that favor few and possibly malicious clients. Theoretical results suggest that our approach provably mitigates backdoor attacks and remains effective over flat loss landscapes. Empirical results on three datasets with different modalities and varying numbers of clients further demonstrate that our approach mitigates a broad class of backdoor attacks with a negligible cost on the model utility.



## **50. Can LLMs Follow Simple Rules?**

cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules;  revised content

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2311.04235v3) [paper-pdf](http://arxiv.org/pdf/2311.04235v3)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Basel Alomair, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.



