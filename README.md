# Latest Adversarial Attack Papers
**update at 2023-06-14 20:40:52**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Class Attribute Inference Attacks: Inferring Sensitive Class Information by Diffusion-Based Attribute Manipulations**

cs.LG

46 pages, 37 figures, 5 tables

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2303.09289v2) [paper-pdf](http://arxiv.org/pdf/2303.09289v2)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Felix Friedrich, Manuel Brack, Patrick Schramowski, Kristian Kersting

**Abstract**: Neural network-based image classifiers are powerful tools for computer vision tasks, but they inadvertently reveal sensitive attribute information about their classes, raising concerns about their privacy. To investigate this privacy leakage, we introduce the first Class Attribute Inference Attack (CAIA), which leverages recent advances in text-to-image synthesis to infer sensitive attributes of individual classes in a black-box setting, while remaining competitive with related white-box attacks. Our extensive experiments in the face recognition domain show that CAIA can accurately infer undisclosed sensitive attributes, such as an individual's hair color, gender, and racial appearance, which are not part of the training labels. Interestingly, we demonstrate that adversarial robust models are even more vulnerable to such privacy leakage than standard models, indicating that a trade-off between robustness and privacy exists.



## **2. Finite Gaussian Neurons: Defending against adversarial attacks by making neural networks say "I don't know"**

cs.LG

PhD thesis

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07796v1) [paper-pdf](http://arxiv.org/pdf/2306.07796v1)

**Authors**: Felix Grezes

**Abstract**: Since 2014, artificial neural networks have been known to be vulnerable to adversarial attacks, which can fool the network into producing wrong or nonsensical outputs by making humanly imperceptible alterations to inputs. While defenses against adversarial attacks have been proposed, they usually involve retraining a new neural network from scratch, a costly task. In this work, I introduce the Finite Gaussian Neuron (FGN), a novel neuron architecture for artificial neural networks. My works aims to: - easily convert existing models to Finite Gaussian Neuron architecture, - while preserving the existing model's behavior on real data, - and offering resistance against adversarial attacks. I show that converted and retrained Finite Gaussian Neural Networks (FGNN) always have lower confidence (i.e., are not overconfident) in their predictions over randomized and Fast Gradient Sign Method adversarial images when compared to classical neural networks, while maintaining high accuracy and confidence over real MNIST images. To further validate the capacity of Finite Gaussian Neurons to protect from adversarial attacks, I compare the behavior of FGNs to that of Bayesian Neural Networks against both randomized and adversarial images, and show how the behavior of the two architectures differs. Finally I show some limitations of the FGN models by testing them on the more complex SPEECHCOMMANDS task, against the stronger Carlini-Wagner and Projected Gradient Descent adversarial attacks.



## **3. Area is all you need: repeatable elements make stronger adversarial attacks**

cs.CV

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07768v1) [paper-pdf](http://arxiv.org/pdf/2306.07768v1)

**Authors**: Dillon Niederhut

**Abstract**: Over the last decade, deep neural networks have achieved state of the art in computer vision tasks. These models, however, are susceptible to unusual inputs, known as adversarial examples, that cause them to misclassify or otherwise fail to detect objects. Here, we provide evidence that the increasing success of adversarial attacks is primarily due to increasing their size. We then demonstrate a method for generating the largest possible adversarial patch by building a adversarial pattern out of repeatable elements. This approach achieves a new state of the art in evading detection by YOLOv2 and YOLOv3. Finally, we present an experiment that fails to replicate the prior success of several attacks published in this field, and end with some comments on testing and reproducibility.



## **4. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

cs.CL

Technical report; 23 pages; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.04528v2) [paper-pdf](http://arxiv.org/pdf/2306.04528v2)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,032 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets, with 567,084 test samples in total. Our findings demonstrate that contemporary LLMs are vulnerable to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. We make our code, prompts, and methodologies to generate adversarial prompts publicly accessible, thereby enabling and encouraging collaborative exploration in this pivotal field: https://github.com/microsoft/promptbench.



## **5. Malafide: a novel adversarial convolutive noise attack against deepfake and spoofing detection systems**

eess.AS

Accepted at INTERSPEECH 2023

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07655v1) [paper-pdf](http://arxiv.org/pdf/2306.07655v1)

**Authors**: Michele Panariello, Wanying Ge, Hemlata Tak, Massimiliano Todisco, Nicholas Evans

**Abstract**: We present Malafide, a universal adversarial attack against automatic speaker verification (ASV) spoofing countermeasures (CMs). By introducing convolutional noise using an optimised linear time-invariant filter, Malafide attacks can be used to compromise CM reliability while preserving other speech attributes such as quality and the speaker's voice. In contrast to other adversarial attacks proposed recently, Malafide filters are optimised independently of the input utterance and duration, are tuned instead to the underlying spoofing attack, and require the optimisation of only a small number of filter coefficients. Even so, they degrade CM performance estimates by an order of magnitude, even in black-box settings, and can also be configured to overcome integrated CM and ASV subsystems. Integrated solutions that use self-supervised learning CMs, however, are more robust, under both black-box and white-box settings.



## **6. A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System**

cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2211.03933v2) [paper-pdf](http://arxiv.org/pdf/2211.03933v2)

**Authors**: Zong-Zhi Lin, Thomas D. Pike, Mark M. Bailey, Nathaniel D. Bastian

**Abstract**: Network intrusion detection systems (NIDS) to detect malicious attacks continue to meet challenges. NIDS are often developed offline while they face auto-generated port scan infiltration attempts, resulting in a significant time lag from adversarial adaption to NIDS response. To address these challenges, we use hypergraphs focused on internet protocol addresses and destination ports to capture evolving patterns of port scan attacks. The derived set of hypergraph-based metrics are then used to train an ensemble machine learning (ML) based NIDS that allows for real-time adaption in monitoring and detecting port scanning activities, other types of attacks, and adversarial intrusions at high accuracy, precision and recall performances. This ML adapting NIDS was developed through the combination of (1) intrusion examples, (2) NIDS update rules, (3) attack threshold choices to trigger NIDS retraining requests, and (4) a production environment with no prior knowledge of the nature of network traffic. 40 scenarios were auto-generated to evaluate the ML ensemble NIDS comprising three tree-based models. The resulting ML Ensemble NIDS was extended and evaluated with the CIC-IDS2017 dataset. Results show that under the model settings of an Update-ALL-NIDS rule (specifically retrain and update all the three models upon the same NIDS retraining request) the proposed ML ensemble NIDS evolved intelligently and produced the best results with nearly 100% detection performance throughout the simulation.



## **7. Extracting Cloud-based Model with Prior Knowledge**

cs.CR

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.04192v4) [paper-pdf](http://arxiv.org/pdf/2306.04192v4)

**Authors**: Shiqian Zhao, Kangjie Chen, Meng Hao, Jian Zhang, Guowen Xu, Hongwei Li, Tianwei Zhang

**Abstract**: Machine Learning-as-a-Service, a pay-as-you-go business pattern, is widely accepted by third-party users and developers. However, the open inference APIs may be utilized by malicious customers to conduct model extraction attacks, i.e., attackers can replicate a cloud-based black-box model merely via querying malicious examples. Existing model extraction attacks mainly depend on the posterior knowledge (i.e., predictions of query samples) from Oracle. Thus, they either require high query overhead to simulate the decision boundary, or suffer from generalization errors and overfitting problems due to query budget limitations. To mitigate it, this work proposes an efficient model extraction attack based on prior knowledge for the first time. The insight is that prior knowledge of unlabeled proxy datasets is conducive to the search for the decision boundary (e.g., informative samples). Specifically, we leverage self-supervised learning including autoencoder and contrastive learning to pre-compile the prior knowledge of the proxy dataset into the feature extractor of the substitute model. Then we adopt entropy to measure and sample the most informative examples to query the target model. Our design leverages both prior and posterior knowledge to extract the model and thus eliminates generalizability errors and overfitting problems. We conduct extensive experiments on open APIs like Traffic Recognition, Flower Recognition, Moderation Recognition, and NSFW Recognition from real-world platforms, Azure and Clarifai. The experimental results demonstrate the effectiveness and efficiency of our attack. For example, our attack achieves 95.1% fidelity with merely 1.8K queries (cost 2.16$) on the NSFW Recognition API. Also, the adversarial examples generated with our substitute model have better transferability than others, which reveals that our scheme is more conducive to downstream attacks.



## **8. I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models**

cs.CV

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07591v1) [paper-pdf](http://arxiv.org/pdf/2306.07591v1)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Modern image-to-text systems typically adopt the encoder-decoder framework, which comprises two main components: an image encoder, responsible for extracting image features, and a transformer-based decoder, used for generating captions. Taking inspiration from the analysis of neural networks' robustness against adversarial perturbations, we propose a novel gray-box algorithm for creating adversarial examples in image-to-text models. Unlike image classification tasks that have a finite set of class labels, finding visually similar adversarial examples in an image-to-text task poses greater challenges because the captioning system allows for a virtually infinite space of possible captions. In this paper, we present a gray-box adversarial attack on image-to-text, both untargeted and targeted. We formulate the process of discovering adversarial perturbations as an optimization problem that uses only the image-encoder component, meaning the proposed attack is language-model agnostic. Through experiments conducted on the ViT-GPT2 model, which is the most-used image-to-text model in Hugging Face, and the Flickr30k dataset, we demonstrate that our proposed attack successfully generates visually similar adversarial examples, both with untargeted and targeted captions. Notably, our attack operates in a gray-box manner, requiring no knowledge about the decoder module. We also show that our attacks fool the popular open-source platform Hugging Face.



## **9. How Secure is Your Website? A Comprehensive Investigation on CAPTCHA Providers and Solving Services**

cs.CR

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07543v1) [paper-pdf](http://arxiv.org/pdf/2306.07543v1)

**Authors**: Rui Jin, Lin Huang, Jikang Duan, Wei Zhao, Yong Liao, Pengyuan Zhou

**Abstract**: Completely Automated Public Turing Test To Tell Computers and Humans Apart (CAPTCHA) has been implemented on many websites to identify between harmful automated bots and legitimate users. However, the revenue generated by the bots has turned circumventing CAPTCHAs into a lucrative business. Although earlier studies provided information about text-based CAPTCHAs and the associated CAPTCHA-solving services, a lot has changed in the past decade regarding content, suppliers, and solvers of CAPTCHA. We have conducted a comprehensive investigation of the latest third-party CAPTCHA providers and CAPTCHA-solving services' attacks. We dug into the details of CAPTCHA-As-a-Service and the latest CAPTCHA-solving services and carried out adversarial experiments on CAPTCHAs and CAPTCHA solvers. The experiment results show a worrying fact: most latest CAPTCHAs are vulnerable to both human solvers and automated solvers. New CAPTCHAs based on hard AI problems and behavior analysis are needed to stop CAPTCHA solvers.



## **10. Adversarial Attacks on the Interpretation of Neuron Activation Maximization**

cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07397v1) [paper-pdf](http://arxiv.org/pdf/2306.07397v1)

**Authors**: Geraldin Nanfack, Alexander Fulleringer, Jonathan Marty, Michael Eickenberg, Eugene Belilovsky

**Abstract**: The internal functional behavior of trained Deep Neural Networks is notoriously difficult to interpret. Activation-maximization approaches are one set of techniques used to interpret and analyze trained deep-learning models. These consist in finding inputs that maximally activate a given neuron or feature map. These inputs can be selected from a data set or obtained by optimization. However, interpretability methods may be subject to being deceived. In this work, we consider the concept of an adversary manipulating a model for the purpose of deceiving the interpretation. We propose an optimization framework for performing this manipulation and demonstrate a number of ways that popular activation-maximization interpretation techniques associated with CNNs can be manipulated to change the interpretations, shedding light on the reliability of these methods.



## **11. Gaussian Membership Inference Privacy**

cs.LG

The first two authors contributed equally

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07273v1) [paper-pdf](http://arxiv.org/pdf/2306.07273v1)

**Authors**: Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci

**Abstract**: We propose a new privacy notion called $f$-Membership Inference Privacy ($f$-MIP), which explicitly considers the capabilities of realistic adversaries under the membership inference attack threat model. By doing so $f$-MIP offers interpretable privacy guarantees and improved utility (e.g., better classification accuracy). Our novel theoretical analysis of likelihood ratio-based membership inference attacks on noisy stochastic gradient descent (SGD) results in a parametric family of $f$-MIP guarantees that we refer to as $\mu$-Gaussian Membership Inference Privacy ($\mu$-GMIP). Our analysis additionally yields an analytical membership inference attack that offers distinct advantages over previous approaches. First, unlike existing methods, our attack does not require training hundreds of shadow models to approximate the likelihood ratio. Second, our analytical attack enables straightforward auditing of our privacy notion $f$-MIP. Finally, our analysis emphasizes the importance of various factors, such as hyperparameters (e.g., batch size, number of model parameters) and data specific characteristics in controlling an attacker's success in reliably inferring a given point's membership to the training set. We demonstrate the effectiveness of our method on models trained across vision and tabular datasets.



## **12. When Vision Fails: Text Attacks Against ViT and OCR**

cs.CR

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07033v1) [paper-pdf](http://arxiv.org/pdf/2306.07033v1)

**Authors**: Nicholas Boucher, Jenny Blessing, Ilia Shumailov, Ross Anderson, Nicolas Papernot

**Abstract**: While text-based machine learning models that operate on visual inputs of rendered text have become robust against a wide range of existing attacks, we show that they are still vulnerable to visual adversarial examples encoded as text. We use the Unicode functionality of combining diacritical marks to manipulate encoded text so that small visual perturbations appear when the text is rendered. We show how a genetic algorithm can be used to generate visual adversarial examples in a black-box setting, and conduct a user study to establish that the model-fooling adversarial examples do not affect human comprehension. We demonstrate the effectiveness of these attacks in the real world by creating adversarial examples against production models published by Facebook, Microsoft, IBM, and Google.



## **13. A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data**

cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2301.10053v2) [paper-pdf](http://arxiv.org/pdf/2301.10053v2)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Andrea Gadotti, Luc Rocher

**Abstract**: Recent advances in synthetic data generation (SDG) have been hailed as a solution to the difficult problem of sharing sensitive data while protecting privacy. SDG aims to learn statistical properties of real data in order to generate "artificial" data that are structurally and statistically similar to sensitive data. However, prior research suggests that inference attacks on synthetic data can undermine privacy, but only for specific outlier records. In this work, we introduce a new attribute inference attack against synthetic data. The attack is based on linear reconstruction methods for aggregate statistics, which target all records in the dataset, not only outliers. We evaluate our attack on state-of-the-art SDG algorithms, including Probabilistic Graphical Models, Generative Adversarial Networks, and recent differentially private SDG mechanisms. By defining a formal privacy game, we show that our attack can be highly accurate even on arbitrary records, and that this is the result of individual information leakage (as opposed to population-level inference). We then systematically evaluate the tradeoff between protecting privacy and preserving statistical utility. Our findings suggest that current SDG methods cannot consistently provide sufficient privacy protection against inference attacks while retaining reasonable utility. The best method evaluated, a differentially private SDG mechanism, can provide both protection against inference attacks and reasonable utility, but only in very specific settings. Lastly, we show that releasing a larger number of synthetic records can improve utility but at the cost of making attacks far more effective.



## **14. How robust accuracy suffers from certified training with convex relaxations**

cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.06995v1) [paper-pdf](http://arxiv.org/pdf/2306.06995v1)

**Authors**: Piersilvio De Bartolomeis, Jacob Clarysse, Amartya Sanyal, Fanny Yang

**Abstract**: Adversarial attacks pose significant threats to deploying state-of-the-art classifiers in safety-critical applications. Two classes of methods have emerged to address this issue: empirical defences and certified defences. Although certified defences come with robustness guarantees, empirical defences such as adversarial training enjoy much higher popularity among practitioners. In this paper, we systematically compare the standard and robust error of these two robust training paradigms across multiple computer vision tasks. We show that in most tasks and for both $\mathscr{l}_\infty$-ball and $\mathscr{l}_2$-ball threat models, certified training with convex relaxations suffers from worse standard and robust error than adversarial training. We further explore how the error gap between certified and adversarial training depends on the threat model and the data distribution. In particular, besides the perturbation budget, we identify as important factors the shape of the perturbation set and the implicit margin of the data distribution. We support our arguments with extensive ablations on both synthetic and image datasets.



## **15. Backdooring Neural Code Search**

cs.SE

Accepted to the 61st Annual Meeting of the Association for  Computational Linguistics (ACL 2023)

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2305.17506v2) [paper-pdf](http://arxiv.org/pdf/2305.17506v2)

**Authors**: Weisong Sun, Yuchen Chen, Guanhong Tao, Chunrong Fang, Xiangyu Zhang, Quanjun Zhang, Bin Luo

**Abstract**: Reusing off-the-shelf code snippets from online repositories is a common practice, which significantly enhances the productivity of software developers. To find desired code snippets, developers resort to code search engines through natural language queries. Neural code search models are hence behind many such engines. These models are based on deep learning and gain substantial attention due to their impressive performance. However, the security aspect of these models is rarely studied. Particularly, an adversary can inject a backdoor in neural code search models, which return buggy or even vulnerable code with security/privacy issues. This may impact the downstream software (e.g., stock trading systems and autonomous driving) and cause financial loss and/or life-threatening incidents. In this paper, we demonstrate such attacks are feasible and can be quite stealthy. By simply modifying one variable/function name, the attacker can make buggy/vulnerable code rank in the top 11%. Our attack BADCODE features a special trigger generation and injection procedure, making the attack more effective and stealthy. The evaluation is conducted on two neural code search models and the results show our attack outperforms baselines by 60%. Our user study demonstrates that our attack is more stealthy than the baseline by two times based on the F1 score.



## **16. Graph Agent Network: Empowering Nodes with Decentralized Communications Capabilities for Adversarial Resilience**

cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.06909v1) [paper-pdf](http://arxiv.org/pdf/2306.06909v1)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Hanyuan Huang, Guangquan Xu, Pan Zhou

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.



## **17. GAN-CAN: A Novel Attack to Behavior-Based Driver Authentication Systems**

cs.CR

16 pages, 6 figures

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.05923v2) [paper-pdf](http://arxiv.org/pdf/2306.05923v2)

**Authors**: Emad Efatinasab, Francesco Marchiori, Denis Donadel, Alessandro Brighente, Mauro Conti

**Abstract**: For many years, car keys have been the sole mean of authentication in vehicles. Whether the access control process is physical or wireless, entrusting the ownership of a vehicle to a single token is prone to stealing attempts. For this reason, many researchers started developing behavior-based authentication systems. By collecting data in a moving vehicle, Deep Learning (DL) models can recognize patterns in the data and identify drivers based on their driving behavior. This can be used as an anti-theft system, as a thief would exhibit a different driving style compared to the vehicle owner's. However, the assumption that an attacker cannot replicate the legitimate driver behavior falls under certain conditions.   In this paper, we propose GAN-CAN, the first attack capable of fooling state-of-the-art behavior-based driver authentication systems in a vehicle. Based on the adversary's knowledge, we propose different GAN-CAN implementations. Our attack leverages the lack of security in the Controller Area Network (CAN) to inject suitably designed time-series data to mimic the legitimate driver. Our design of the malicious time series results from the combination of different Generative Adversarial Networks (GANs) and our study on the safety importance of the injected values during the attack. We tested GAN-CAN in an improved version of the most efficient driver behavior-based authentication model in the literature. We prove that our attack can fool it with an attack success rate of up to 0.99. We show how an attacker, without prior knowledge of the authentication system, can steal a car by deploying GAN-CAN in an off-the-shelf system in under 22 minutes.



## **18. Asymptotically Optimal Adversarial Strategies for the Probability Estimation Framework**

quant-ph

54 pages

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06802v1) [paper-pdf](http://arxiv.org/pdf/2306.06802v1)

**Authors**: Soumyadip Patra, Peter Bierhorst

**Abstract**: The Probability Estimation Framework involves direct estimation of the probability of occurrences of outcomes conditioned on measurement settings and side information. It is a powerful tool for certifying randomness in quantum non-locality experiments. In this paper, we present a self-contained proof of the asymptotic optimality of the method. Our approach refines earlier results to allow a better characterisation of optimal adversarial attacks on the protocol. We apply these results to the (2,2,2) Bell scenario, obtaining an analytic characterisation of the optimal adversarial attacks bound by no-signalling principles, while also demonstrating the asymptotic robustness of the PEF method to deviations from expected experimental behaviour. We also study extensions of the analysis to quantum-limited adversaries in the (2,2,2) Bell scenario and no-signalling adversaries in higher $(n,m,k)$ Bell scenarios.



## **19. Adversarial Reconnaissance Mitigation and Modeling**

cs.CR

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06769v1) [paper-pdf](http://arxiv.org/pdf/2306.06769v1)

**Authors**: Shanto Roy, Nazia Sharmin, Mohammad Sujan Miah, Jaime C Acosta, Christopher Kiekintveld, Aron Laszka

**Abstract**: Adversarial reconnaissance is a crucial step in sophisticated cyber-attacks as it enables threat actors to find the weakest points of otherwise well-defended systems. To thwart reconnaissance, defenders can employ cyber deception techniques, such as deploying honeypots. In recent years, researchers have made great strides in developing game-theoretic models to find optimal deception strategies. However, most of these game-theoretic models build on relatively simple models of adversarial reconnaissance -- even though reconnaissance should be a focus point as the very purpose of deception is to thwart reconnaissance. In this paper, we first discuss effective cyber reconnaissance mitigation techniques including deception strategies and beyond. Then we provide a review of the literature on deception games from the perspective of modeling adversarial reconnaissance, highlighting key aspects of reconnaissance that have not been adequately captured in prior work. We then describe a probability-theory based model of the adversaries' belief formation and illustrate using numerical examples that this model can capture key aspects of adversarial reconnaissance. We believe that our review and belief model can serve as a stepping stone for developing more realistic and practical deception games.



## **20. Neural Architecture Design and Robustness: A Dataset**

cs.LG

ICLR 2023; project page: http://robustness.vision/

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06712v1) [paper-pdf](http://arxiv.org/pdf/2306.06712v1)

**Authors**: Steffen Jung, Jovita Lukasik, Margret Keuper

**Abstract**: Deep learning models have proven to be successful in a wide range of machine learning tasks. Yet, they are often highly sensitive to perturbations on the input data which can lead to incorrect decisions with high confidence, hampering their deployment for practical use-cases. Thus, finding architectures that are (more) robust against perturbations has received much attention in recent years. Just like the search for well-performing architectures in terms of clean accuracy, this usually involves a tedious trial-and-error process with one additional challenge: the evaluation of a network's robustness is significantly more expensive than its evaluation for clean accuracy. Thus, the aim of this paper is to facilitate better streamlined research on architectural design choices with respect to their impact on robustness as well as, for example, the evaluation of surrogate measures for robustness. We therefore borrow one of the most commonly considered search spaces for neural architecture search for image classification, NAS-Bench-201, which contains a manageable size of 6466 non-isomorphic network designs. We evaluate all these networks on a range of common adversarial attacks and corruption types and introduce a database on neural architecture design and robustness evaluations. We further present three exemplary use cases of this dataset, in which we (i) benchmark robustness measurements based on Jacobian and Hessian matrices for their robustness predictability, (ii) perform neural architecture search on robust accuracies, and (iii) provide an initial analysis of how architectural design choices affect robustness. We find that carefully crafting the topology of a network can have substantial impact on its robustness, where networks with the same parameter count range in mean adversarial robust accuracy from 20%-41%. Code and data is available at http://robustness.vision/.



## **21. EvadeDroid: A Practical Evasion Attack on Machine Learning for Black-box Android Malware Detection**

cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2110.03301v3) [paper-pdf](http://arxiv.org/pdf/2110.03301v3)

**Authors**: Hamid Bostani, Veelasha Moonsamy

**Abstract**: Over the last decade, researchers have extensively explored the vulnerabilities of Android malware detectors to adversarial examples through the development of evasion attacks; however, the practicality of these attacks in real-world scenarios remains arguable. The majority of studies have assumed attackers know the details of the target classifiers used for malware detection, while in reality, malicious actors have limited access to the target classifiers. This paper introduces EvadeDroid, a practical decision-based adversarial attack designed to effectively evade black-box Android malware detectors in real-world scenarios. In addition to generating real-world adversarial malware, the proposed evasion attack can also preserve the functionality of the original malware applications (apps). EvadeDroid constructs a collection of functionality-preserving transformations derived from benign donors that share opcode-level similarity with malware apps by leveraging an n-gram-based approach. These transformations are then used to morph malware instances into benign ones via an iterative and incremental manipulation strategy. The proposed manipulation technique is a novel, query-efficient optimization algorithm that can find and inject optimal sequences of transformations into malware apps. Our empirical evaluation demonstrates the efficacy of EvadeDroid under soft- and hard-label attacks. Furthermore, EvadeDroid exhibits the capability to generate real-world adversarial examples that can effectively evade a wide range of black-box ML-based malware detectors with minimal query requirements. Finally, we show that the proposed problem-space adversarial attack is able to preserve its stealthiness against five popular commercial antiviruses, thus demonstrating its feasibility in the real world.



## **22. Level Up with RealAEs: Leveraging Domain Constraints in Feature Space to Strengthen Robustness of Android Malware Detection**

cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2205.15128v3) [paper-pdf](http://arxiv.org/pdf/2205.15128v3)

**Authors**: Hamid Bostani, Zhengyu Zhao, Zhuoran Liu, Veelasha Moonsamy

**Abstract**: The vulnerability to adversarial examples remains one major obstacle for Machine Learning (ML)-based Android malware detection. Realistic attacks in the Android malware domain create Realizable Adversarial Examples (RealAEs), i.e., AEs that satisfy the domain constraints of Android malware. Recent studies have shown that using such RealAEs in Adversarial Training (AT) is more effective in defending against realistic attacks than using unrealizable AEs (unRealAEs). This is because RealAEs allow defenders to explore certain pockets in the feature space that are vulnerable to realistic attacks. However, existing defenses commonly generate RealAEs in the problem space, which is known to be time-consuming and impractical for AT. In this paper, we propose to generate RealAEs in the feature space, leading to a simpler and more efficient solution. Our approach is driven by a novel interpretation of Android domain constraints in the feature space. More concretely, our defense first learns feature-space domain constraints by extracting meaningful feature dependencies from data and then applies them to generating feature-space RealAEs during AT. Extensive experiments on DREBIN, a well-known Android malware detector, demonstrate that our new defense outperforms not only unRealAE-based AT but also the state-of-the-art defense that relies on non-uniform perturbations. We further validate the ability of our learned feature-space domain constraints in representing Android malware properties by showing that our feature-space domain constraints can help distinguish RealAEs from unRealAEs.



## **23. Attacking Cooperative Multi-Agent Reinforcement Learning by Adversarial Minority Influence**

cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2302.03322v2) [paper-pdf](http://arxiv.org/pdf/2302.03322v2)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Pu Feng, Xin Yu, Aishan Liu, Wenjun Wu, Xianglong Liu

**Abstract**: This study probes the vulnerabilities of cooperative multi-agent reinforcement learning (c-MARL) under adversarial attacks, a critical determinant of c-MARL's worst-case performance prior to real-world implementation. Current observation-based attacks, constrained by white-box assumptions, overlook c-MARL's complex multi-agent interactions and cooperative objectives, resulting in impractical and limited attack capabilities. To address these shortcomes, we propose Adversarial Minority Influence (AMI), a practical and strong for c-MARL. AMI is a practical black-box attack and can be launched without knowing victim parameters. AMI is also strong by considering the complex multi-agent interaction and the cooperative goal of agents, enabling a single adversarial agent to unilaterally misleads majority victims to form targeted worst-case cooperation. This mirrors minority influence phenomena in social psychology. To achieve maximum deviation in victim policies under complex agent-wise interactions, our unilateral attack aims to characterize and maximize the impact of the adversary on the victims. This is achieved by adapting a unilateral agent-wise relation metric derived from mutual information, thereby mitigating the adverse effects of victim influence on the adversary. To lead the victims into a jointly detrimental scenario, our targeted attack deceives victims into a long-term, cooperatively harmful situation by guiding each victim towards a specific target, determined through a trial-and-error process executed by a reinforcement learning agent. Through AMI, we achieve the first successful attack against real-world robot swarms and effectively fool agents in simulated environments into collectively worst-case scenarios, including Starcraft II and Multi-agent Mujoco. The source code and demonstrations can be found at: https://github.com/DIG-Beihang/AMI.



## **24. Defense Against Adversarial Attacks on Audio DeepFake Detection**

cs.SD

Accepted to INTERSPEECH 2023

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2212.14597v2) [paper-pdf](http://arxiv.org/pdf/2212.14597v2)

**Authors**: Piotr Kawa, Marcin Plata, Piotr Syga

**Abstract**: Audio DeepFakes (DF) are artificially generated utterances created using deep learning, with the primary aim of fooling the listeners in a highly convincing manner. Their quality is sufficient to pose a severe threat in terms of security and privacy, including the reliability of news or defamation. Multiple neural network-based methods to detect generated speech have been proposed to prevent the threats. In this work, we cover the topic of adversarial attacks, which decrease the performance of detectors by adding superficial (difficult to spot by a human) changes to input data. Our contribution contains evaluating the robustness of 3 detection architectures against adversarial attacks in two scenarios (white-box and using transferability) and enhancing it later by using adversarial training performed by our novel adaptive training. Moreover, one of the investigated architectures is RawNet3, which, to the best of our knowledge, we adapted for the first time to DeepFake detection.



## **25. The Defense of Networked Targets in General Lotto games**

cs.GT

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06485v1) [paper-pdf](http://arxiv.org/pdf/2306.06485v1)

**Authors**: Adel Aghajan, Keith Paarporn, Jason R. Marden

**Abstract**: Ensuring the security of networked systems is a significant problem, considering the susceptibility of modern infrastructures and technologies to adversarial interference. A central component of this problem is how defensive resources should be allocated to mitigate the severity of potential attacks on the system. In this paper, we consider this in the context of a General Lotto game, where a defender and attacker deploys resources on the nodes of a network, and the objective is to secure as many links as possible. The defender secures a link only if it out-competes the attacker on both of its associated nodes. For bipartite networks, we completely characterize equilibrium payoffs and strategies for both the defender and attacker. Surprisingly, the resulting payoffs are the same for any bipartite graph. On arbitrary network structures, we provide lower and upper bounds on the defender's max-min value. Notably, the equilibrium payoff from bipartite networks serves as the lower bound. These results suggest that more connected networks are easier to defend against attacks. We confirm these findings with simulations that compute deterministic allocation strategies on large random networks. This also highlights the importance of randomization in the equilibrium strategies.



## **26. NeRFool: Uncovering the Vulnerability of Generalizable Neural Radiance Fields against Adversarial Perturbations**

cs.CV

Accepted by ICML 2023

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06359v1) [paper-pdf](http://arxiv.org/pdf/2306.06359v1)

**Authors**: Yonggan Fu, Ye Yuan, Souvik Kundu, Shang Wu, Shunyao Zhang, Yingyan Lin

**Abstract**: Generalizable Neural Radiance Fields (GNeRF) are one of the most promising real-world solutions for novel view synthesis, thanks to their cross-scene generalization capability and thus the possibility of instant rendering on new scenes. While adversarial robustness is essential for real-world applications, little study has been devoted to understanding its implication on GNeRF. We hypothesize that because GNeRF is implemented by conditioning on the source views from new scenes, which are often acquired from the Internet or third-party providers, there are potential new security concerns regarding its real-world applications. Meanwhile, existing understanding and solutions for neural networks' adversarial robustness may not be applicable to GNeRF, due to its 3D nature and uniquely diverse operations. To this end, we present NeRFool, which to the best of our knowledge is the first work that sets out to understand the adversarial robustness of GNeRF. Specifically, NeRFool unveils the vulnerability patterns and important insights regarding GNeRF's adversarial robustness. Built upon the above insights gained from NeRFool, we further develop NeRFool+, which integrates two techniques capable of effectively attacking GNeRF across a wide range of target views, and provide guidelines for defending against our proposed attacks. We believe that our NeRFool/NeRFool+ lays the initial foundation for future innovations in developing robust real-world GNeRF solutions. Our codes are available at: https://github.com/GATECH-EIC/NeRFool.



## **27. Differentially private sliced inverse regression in the federated paradigm**

stat.ME

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06324v1) [paper-pdf](http://arxiv.org/pdf/2306.06324v1)

**Authors**: Shuaida He, Jiarui Zhang, Xin Chen

**Abstract**: We extend the celebrated sliced inverse regression to address the challenges of decentralized data, prioritizing privacy and communication efficiency. Our approach, federated sliced inverse regression (FSIR), facilitates collaborative estimation of the sufficient dimension reduction subspace among multiple clients, solely sharing local estimates to protect sensitive datasets from exposure. To guard against potential adversary attacks, FSIR further employs diverse perturbation strategies, including a novel multivariate Gaussian mechanism that guarantees differential privacy at a low cost of statistical accuracy. Additionally, FSIR naturally incorporates a collaborative variable screening step, enabling effective handling of high-dimensional client data. Theoretical properties of FSIR are established for both low-dimensional and high-dimensional settings, supported by extensive numerical experiments and real data analysis.



## **28. The Certification Paradox: Certifications Admit Better Attacks**

cs.LG

16 pages, 6 figures

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2302.04379v2) [paper-pdf](http://arxiv.org/pdf/2302.04379v2)

**Authors**: Andrew C. Cullen, Shijie Liu, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In guaranteeing that no adversarial examples exist within a bounded region, certification mechanisms play an important role in demonstrating the robustness of neural networks. In this work we ask: Could certifications have any unintended consequences, through exposing additional information about certified models? We answer this question in the affirmative, demonstrating that certifications not only measure model robustness but also present a new attack surface. We propose \emph{Certification Aware Attacks}, that produce smaller adversarial perturbations more than twice as frequently as any prior approach, when launched against certified models. Our attacks achieve an up to $34\%$ reduction in the median perturbation norm (comparing target and attack instances), while requiring $90 \%$ less computational time than approaches like PGD. That our attacks achieve such significant reductions in perturbation size and computational cost highlights an apparent paradox in deploying certification mechanisms. We end the paper with a discussion of how these risks could potentially be mitigated.



## **29. Divide and Repair: Using Options to Improve Performance of Imitation Learning Against Adversarial Demonstrations**

cs.LG

33 pages, 4 figures, 3 tables

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.04581v2) [paper-pdf](http://arxiv.org/pdf/2306.04581v2)

**Authors**: Prithviraj Dasgupta

**Abstract**: We consider the problem of learning to perform a task from demonstrations given by teachers or experts, when some of the experts' demonstrations might be adversarial and demonstrate an incorrect way to perform the task. We propose a novel technique that can identify parts of demonstrated trajectories that have not been significantly modified by the adversary and utilize them for learning, using temporally extended policies or options. We first define a trajectory divergence measure based on the spatial and temporal features of demonstrated trajectories to detect and discard parts of the trajectories that have been significantly modified by an adversarial expert, and, could degrade the learner's performance, if used for learning, We then use an options-based algorithm that partitions trajectories and learns only from the parts of trajectories that have been determined as admissible. We provide theoretical results of our technique to show that repairing partial trajectories improves the sample efficiency of the demonstrations without degrading the learner's performance. We then evaluate the proposed algorithm for learning to play an Atari-like, computer-based game called LunarLander in the presence of different types and degrees of adversarial attacks of demonstrated trajectories. Our experimental results show that our technique can identify adversarially modified parts of the demonstrated trajectories and successfully prevent the learning performance from degrading due to adversarial demonstrations.



## **30. Overcoming Adversarial Attacks for Human-in-the-Loop Applications**

cs.LG

New Frontiers in Adversarial Machine Learning, ICML 2022

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05952v1) [paper-pdf](http://arxiv.org/pdf/2306.05952v1)

**Authors**: Ryan McCoppin, Marla Kennedy, Platon Lukyanenko, Sean Kennedy

**Abstract**: Including human analysis has the potential to positively affect the robustness of Deep Neural Networks and is relatively unexplored in the Adversarial Machine Learning literature. Neural network visual explanation maps have been shown to be prone to adversarial attacks. Further research is needed in order to select robust visualizations of explanations for the image analyst to evaluate a given model. These factors greatly impact Human-In-The-Loop (HITL) evaluation tools due to their reliance on adversarial images, including explanation maps and measurements of robustness. We believe models of human visual attention may improve interpretability and robustness of human-machine imagery analysis systems. Our challenge remains, how can HITL evaluation be robust in this adversarial landscape?



## **31. Detecting Adversarial Directions in Deep Reinforcement Learning to Make Robust Decisions**

cs.LG

Published in ICML 2023

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05873v1) [paper-pdf](http://arxiv.org/pdf/2306.05873v1)

**Authors**: Ezgi Korkmaz, Jonah Brown-Cohen

**Abstract**: Learning in MDPs with highly complex state representations is currently possible due to multiple advancements in reinforcement learning algorithm design. However, this incline in complexity, and furthermore the increase in the dimensions of the observation came at the cost of volatility that can be taken advantage of via adversarial attacks (i.e. moving along worst-case directions in the observation space). To solve this policy instability problem we propose a novel method to detect the presence of these non-robust directions via local quadratic approximation of the deep neural policy loss. Our method provides a theoretical basis for the fundamental cut-off between safe observations and adversarial observations. Furthermore, our technique is computationally efficient, and does not depend on the methods used to produce the worst-case directions. We conduct extensive experiments in the Arcade Learning Environment with several different adversarial attack techniques. Most significantly, we demonstrate the effectiveness of our approach even in the setting where non-robust directions are explicitly optimized to circumvent our proposed method.



## **32. Towards a Robust Detection of Language Model Generated Text: Is ChatGPT that Easy to Detect?**

cs.CL

Accepted to TALN 2023

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05871v1) [paper-pdf](http://arxiv.org/pdf/2306.05871v1)

**Authors**: Wissam Antoun, Virginie Mouilleron, Benoît Sagot, Djamé Seddah

**Abstract**: Recent advances in natural language processing (NLP) have led to the development of large language models (LLMs) such as ChatGPT. This paper proposes a methodology for developing and evaluating ChatGPT detectors for French text, with a focus on investigating their robustness on out-of-domain data and against common attack schemes. The proposed method involves translating an English dataset into French and training a classifier on the translated data. Results show that the detectors can effectively detect ChatGPT-generated text, with a degree of robustness against basic attack techniques in in-domain settings. However, vulnerabilities are evident in out-of-domain contexts, highlighting the challenge of detecting adversarial text. The study emphasizes caution when applying in-domain testing results to a wider variety of content. We provide our translated datasets and models as open-source resources. https://gitlab.inria.fr/wantoun/robust-chatgpt-detection



## **33. COVER: A Heuristic Greedy Adversarial Attack on Prompt-based Learning in Language Models**

cs.CL

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05659v1) [paper-pdf](http://arxiv.org/pdf/2306.05659v1)

**Authors**: Zihao Tan, Qingliang Chen, Wenbin Zhu, Yongjian Huang

**Abstract**: Prompt-based learning has been proved to be an effective way in pre-trained language models (PLMs), especially in low-resource scenarios like few-shot settings. However, the trustworthiness of PLMs is of paramount significance and potential vulnerabilities have been shown in prompt-based templates that could mislead the predictions of language models, causing serious security concerns. In this paper, we will shed light on some vulnerabilities of PLMs, by proposing a prompt-based adversarial attack on manual templates in black box scenarios. First of all, we design character-level and word-level heuristic approaches to break manual templates separately. Then we present a greedy algorithm for the attack based on the above heuristic destructive approaches. Finally, we evaluate our approach with the classification tasks on three variants of BERT series models and eight datasets. And comprehensive experimental results justify the effectiveness of our approach in terms of attack success rate and attack speed. Further experimental studies indicate that our proposed method also displays good capabilities in scenarios with varying shot counts, template lengths and query counts, exhibiting good generalizability.



## **34. Spike timing reshapes robustness against attacks in spiking neural networks**

q-bio.NC

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05654v1) [paper-pdf](http://arxiv.org/pdf/2306.05654v1)

**Authors**: Jianhao Ding, Zhaofei Yu, Tiejun Huang, Jian K. Liu

**Abstract**: The success of deep learning in the past decade is partially shrouded in the shadow of adversarial attacks. In contrast, the brain is far more robust at complex cognitive tasks. Utilizing the advantage that neurons in the brain communicate via spikes, spiking neural networks (SNNs) are emerging as a new type of neural network model, boosting the frontier of theoretical investigation and empirical application of artificial neural networks and deep learning. Neuroscience research proposes that the precise timing of neural spikes plays an important role in the information coding and sensory processing of the biological brain. However, the role of spike timing in SNNs is less considered and far from understood. Here we systematically explored the timing mechanism of spike coding in SNNs, focusing on the robustness of the system against various types of attacks. We found that SNNs can achieve higher robustness improvement using the coding principle of precise spike timing in neural encoding and decoding, facilitated by different learning rules. Our results suggest that the utility of spike timing coding in SNNs could improve the robustness against attacks, providing a new approach to reliable coding principles for developing next-generation brain-inspired deep learning.



## **35. McFIL: Model Counting Functionality-Inherent Leakage**

cs.CR

To appear in USENIX Security 2023

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05633v1) [paper-pdf](http://arxiv.org/pdf/2306.05633v1)

**Authors**: Maximilian Zinkus, Yinzhi Cao, Matthew Green

**Abstract**: Protecting the confidentiality of private data and using it for useful collaboration have long been at odds. Modern cryptography is bridging this gap through rapid growth in secure protocols such as multi-party computation, fully-homomorphic encryption, and zero-knowledge proofs. However, even with provable indistinguishability or zero-knowledgeness, confidentiality loss from leakage inherent to the functionality may partially or even completely compromise secret values without ever falsifying proofs of security. In this work, we describe McFIL, an algorithmic approach and accompanying software implementation which automatically quantifies intrinsic leakage for a given functionality. Extending and generalizing the Chosen-Ciphertext attack framework of Beck et al. with a practical heuristic, our approach not only quantifies but maximizes functionality-inherent leakage using Maximum Model Counting within a SAT solver. As a result, McFIL automatically derives approximately-optimal adversary inputs that, when used in secure protocols, maximize information leakage of private values.



## **36. Robustness Testing for Multi-Agent Reinforcement Learning: State Perturbations on Critical Agents**

cs.LG

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.06136v1) [paper-pdf](http://arxiv.org/pdf/2306.06136v1)

**Authors**: Ziyuan Zhou, Guanjun Liu

**Abstract**: Multi-Agent Reinforcement Learning (MARL) has been widely applied in many fields such as smart traffic and unmanned aerial vehicles. However, most MARL algorithms are vulnerable to adversarial perturbations on agent states. Robustness testing for a trained model is an essential step for confirming the trustworthiness of the model against unexpected perturbations. This work proposes a novel Robustness Testing framework for MARL that attacks states of Critical Agents (RTCA). The RTCA has two innovations: 1) a Differential Evolution (DE) based method to select critical agents as victims and to advise the worst-case joint actions on them; and 2) a team cooperation policy evaluation method employed as the objective function for the optimization of DE. Then, adversarial state perturbations of the critical agents are generated based on the worst-case joint actions. This is the first robustness testing framework with varying victim agents. RTCA demonstrates outstanding performance in terms of the number of victim agents and destroying cooperation policies.



## **37. Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning**

cs.CR

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.05494v1) [paper-pdf](http://arxiv.org/pdf/2306.05494v1)

**Authors**: Mohamed el Shehaby, Ashraf Matrawy

**Abstract**: Machine Learning (ML) has become ubiquitous, and its deployment in Network Intrusion Detection Systems (NIDS) is inevitable due to its automated nature and high accuracy in processing and classifying large volumes of data. However, ML has been found to have several flaws, on top of them are adversarial attacks, which aim to trick ML models into producing faulty predictions. While most adversarial attack research focuses on computer vision datasets, recent studies have explored the practicality of such attacks against ML-based network security entities, especially NIDS.   This paper presents two distinct contributions: a taxonomy of practicality issues associated with adversarial attacks against ML-based NIDS and an investigation of the impact of continuous training on adversarial attacks against NIDS. Our experiments indicate that continuous re-training, even without adversarial training, can reduce the effect of adversarial attacks. While adversarial attacks can harm ML-based NIDSs, our aim is to highlight that there is a significant gap between research and real-world practicality in this domain which requires attention.



## **38. Ownership Protection of Generative Adversarial Networks**

cs.CR

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.05233v1) [paper-pdf](http://arxiv.org/pdf/2306.05233v1)

**Authors**: Hailong Hu, Jun Pang

**Abstract**: Generative adversarial networks (GANs) have shown remarkable success in image synthesis, making GAN models themselves commercially valuable to legitimate model owners. Therefore, it is critical to technically protect the intellectual property of GANs. Prior works need to tamper with the training set or training process, and they are not robust to emerging model extraction attacks. In this paper, we propose a new ownership protection method based on the common characteristics of a target model and its stolen models. Our method can be directly applicable to all well-trained GANs as it does not require retraining target models. Extensive experimental results show that our new method can achieve the best protection performance, compared to the state-of-the-art methods. Finally, we demonstrate the effectiveness of our method with respect to the number of generations of model extraction attacks, the number of generated samples, different datasets, as well as adaptive attacks.



## **39. Boosting Adversarial Transferability by Achieving Flat Local Maxima**

cs.CV

17 pages, 5 figures, 6 tables

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.05225v1) [paper-pdf](http://arxiv.org/pdf/2306.05225v1)

**Authors**: Zhijin Ge, Fanhua Shang, Hongying Liu, Yuanyuan Liu, Xiaosen Wang

**Abstract**: Transfer-based attack adopts the adversarial examples generated on the surrogate model to attack various models, making it applicable in the physical world and attracting increasing interest. Recently, various adversarial attacks have emerged to boost adversarial transferability from different perspectives. In this work, inspired by the fact that flat local minima are correlated with good generalization, we assume and empirically validate that adversarial examples at a flat local region tend to have good transferability by introducing a penalized gradient norm to the original loss function. Since directly optimizing the gradient regularization norm is computationally expensive and intractable for generating adversarial examples, we propose an approximation optimization method to simplify the gradient update of the objective function. Specifically, we randomly sample an example and adopt the first-order gradient to approximate the second-order Hessian matrix, which makes computing more efficient by interpolating two Jacobian matrices. Meanwhile, in order to obtain a more stable gradient direction, we randomly sample multiple examples and average the gradients of these examples to reduce the variance due to random sampling during the iterative process. Extensive experimental results on the ImageNet-compatible dataset show that the proposed method can generate adversarial examples at flat local regions, and significantly improve the adversarial transferability on either normally trained models or adversarially trained models than the state-of-the-art attacks.



## **40. PriSampler: Mitigating Property Inference of Diffusion Models**

cs.CR

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.05208v1) [paper-pdf](http://arxiv.org/pdf/2306.05208v1)

**Authors**: Hailong Hu, Jun Pang

**Abstract**: Diffusion models have been remarkably successful in data synthesis. Such successes have also driven diffusion models to apply to sensitive data, such as human face data, but this might bring about severe privacy concerns. In this work, we systematically present the first privacy study about property inference attacks against diffusion models, in which adversaries aim to extract sensitive global properties of the training set from a diffusion model, such as the proportion of the training data for certain sensitive properties. Specifically, we consider the most practical attack scenario: adversaries are only allowed to obtain synthetic data. Under this realistic scenario, we evaluate the property inference attacks on different types of samplers and diffusion models. A broad range of evaluations shows that various diffusion models and their samplers are all vulnerable to property inference attacks. Furthermore, one case study on off-the-shelf pre-trained diffusion models also demonstrates the effectiveness of the attack in practice. Finally, we propose a new model-agnostic plug-in method PriSampler to mitigate the property inference of diffusion models. PriSampler can be directly applied to well-trained diffusion models and support both stochastic and deterministic sampling. Extensive experiments illustrate the effectiveness of our defense and it makes adversaries infer the proportion of properties as close as random guesses. PriSampler also shows its significantly superior performance to diffusion models trained with differential privacy on both model utility and defense performance.



## **41. Towards Robust Neural Image Compression: Adversarial Attack and Model Finetuning**

cs.CV

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2112.08691v3) [paper-pdf](http://arxiv.org/pdf/2112.08691v3)

**Authors**: Tong Chen, Zhan Ma

**Abstract**: Deep neural network-based image compression has been extensively studied. However, the model robustness which is crucial to practical application is largely overlooked. We propose to examine the robustness of prevailing learned image compression models by injecting negligible adversarial perturbation into the original source image. Severe distortion in decoded reconstruction reveals the general vulnerability in existing methods regardless of their settings (e.g., network architecture, loss function, quality scale). A variety of defense strategies including geometric self-ensemble based pre-processing, and adversarial training, are investigated against the adversarial attack to improve the model's robustness. Later the defense efficiency is further exemplified in real-life image recompression case studies. Overall, our methodology is simple, effective, and generalizable, making it attractive for developing robust learned image compression solutions. All materials are made publicly accessible at https://njuvision.github.io/RobustNIC for reproducible research.



## **42. Adversarial Sample Detection Through Neural Network Transport Dynamics**

cs.LG

ECML PKDD 2023

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04252v2) [paper-pdf](http://arxiv.org/pdf/2306.04252v2)

**Authors**: Skander Karkar, Patrick Gallinari, Alain Rakotomamonjy

**Abstract**: We propose a detector of adversarial samples that is based on the view of neural networks as discrete dynamic systems. The detector tells clean inputs from abnormal ones by comparing the discrete vector fields they follow through the layers. We also show that regularizing this vector field during training makes the network more regular on the data distribution's support, thus making the activations of clean inputs more distinguishable from those of abnormal ones. Experimentally, we compare our detector favorably to other detectors on seen and unseen attacks, and show that the regularization of the network's dynamics improves the performance of adversarial detectors that use the internal embeddings as inputs, while also improving test accuracy.



## **43. Toward Enhanced Robustness in Unsupervised Graph Representation Learning: A Graph Information Bottleneck Perspective**

cs.LG

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2201.08557v2) [paper-pdf](http://arxiv.org/pdf/2201.08557v2)

**Authors**: Jihong Wang, Minnan Luo, Jundong Li, Ziqi Liu, Jun Zhou, Qinghua Zheng

**Abstract**: Recent studies have revealed that GNNs are vulnerable to adversarial attacks. Most existing robust graph learning methods measure model robustness based on label information, rendering them infeasible when label information is not available. A straightforward direction is to employ the widely used Infomax technique from typical Unsupervised Graph Representation Learning (UGRL) to learn robust unsupervised representations. Nonetheless, directly transplanting the Infomax technique from typical UGRL to robust UGRL may involve a biased assumption. In light of the limitation of Infomax, we propose a novel unbiased robust UGRL method called Robust Graph Information Bottleneck (RGIB), which is grounded in the Information Bottleneck (IB) principle. Our RGIB attempts to learn robust node representations against adversarial perturbations by preserving the original information in the benign graph while eliminating the adversarial information in the adversarial graph. There are mainly two challenges to optimize RGIB: 1) high complexity of adversarial attack to perturb node features and graph structure jointly in the training procedure; 2) mutual information estimation upon adversarially attacked graphs. To tackle these problems, we further propose an efficient adversarial training strategy with only feature perturbations and an effective mutual information estimator with subgraph-level summary. Moreover, we theoretically establish a connection between our proposed RGIB and the robustness of downstream classifiers, revealing that RGIB can provide a lower bound on the adversarial risk of downstream classifiers. Extensive experiments over several benchmarks and downstream tasks demonstrate the effectiveness and superiority of our proposed method.



## **44. Generalizable Lightweight Proxy for Robust NAS against Diverse Perturbations**

cs.LG

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.05031v1) [paper-pdf](http://arxiv.org/pdf/2306.05031v1)

**Authors**: Hyeonjeong Ha, Minseon Kim, Sung Ju Hwang

**Abstract**: Recent neural architecture search (NAS) frameworks have been successful in finding optimal architectures for given conditions (e.g., performance or latency). However, they search for optimal architectures in terms of their performance on clean images only, while robustness against various types of perturbations or corruptions is crucial in practice. Although there exist several robust NAS frameworks that tackle this issue by integrating adversarial training into one-shot NAS, however, they are limited in that they only consider robustness against adversarial attacks and require significant computational resources to discover optimal architectures for a single task, which makes them impractical in real-world scenarios. To address these challenges, we propose a novel lightweight robust zero-cost proxy that considers the consistency across features, parameters, and gradients of both clean and perturbed images at the initialization state. Our approach facilitates an efficient and rapid search for neural architectures capable of learning generalizable features that exhibit robustness across diverse perturbations. The experimental results demonstrate that our proxy can rapidly and efficiently search for neural architectures that are consistently robust against various perturbations on multiple benchmark datasets and diverse search spaces, largely outperforming existing clean zero-shot NAS and robust NAS with reduced search cost.



## **45. A Melting Pot of Evolution and Learning**

cs.NE

To Appear in Proceedings of Genetic Programming Theory & Practice XX,  2023

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04971v1) [paper-pdf](http://arxiv.org/pdf/2306.04971v1)

**Authors**: Moshe Sipper, Achiya Elyasaf, Tomer Halperin, Zvika Haramaty, Raz Lapid, Eyal Segal, Itai Tzruia, Snir Vitrack Tamam

**Abstract**: We survey eight recent works by our group, involving the successful blending of evolutionary algorithms with machine learning and deep learning: 1. Binary and Multinomial Classification through Evolutionary Symbolic Regression, 2. Classy Ensemble: A Novel Ensemble Algorithm for Classification, 3. EC-KitY: Evolutionary Computation Tool Kit in Python, 4. Evolution of Activation Functions for Deep Learning-Based Image Classification, 5. Adaptive Combination of a Genetic Algorithm and Novelty Search for Deep Neuroevolution, 6. An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Networks, 7. Foiling Explanations in Deep Neural Networks, 8. Patch of Invisibility: Naturalistic Black-Box Adversarial Attacks on Object Detectors.



## **46. FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and LLMs**

cs.CR

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04959v1) [paper-pdf](http://arxiv.org/pdf/2306.04959v1)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Chulin Xie, Kai Zhang, Qifan Zhang, Yuhui Zhang, Chaoyang He, Salman Avestimehr

**Abstract**: This paper introduces FedMLSecurity, a benchmark that simulates adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). As an integral module of the open-sourced library FedML that facilitates FL algorithm development and performance comparison, FedMLSecurity enhances the security assessment capacity of FedML. FedMLSecurity comprises two principal components: FedMLAttacker, which simulates attacks injected into FL training, and FedMLDefender, which emulates defensive strategies designed to mitigate the impacts of the attacks. FedMLSecurity is open-sourced 1 and is customizable to a wide range of machine learning models (e.g., Logistic Regression, ResNet, GAN, etc.) and federated optimizers (e.g., FedAVG, FedOPT, FedNOVA, etc.). Experimental evaluations in this paper also demonstrate the ease of application of FedMLSecurity to Large Language Models (LLMs), further reinforcing its versatility and practical utility in various scenarios.



## **47. Degraded Polygons Raise Fundamental Questions of Neural Network Perception**

cs.CV

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04955v1) [paper-pdf](http://arxiv.org/pdf/2306.04955v1)

**Authors**: Leonard Tang, Dan Ley

**Abstract**: It is well-known that modern computer vision systems often exhibit behaviors misaligned with those of humans: from adversarial attacks to image corruptions, deep learning vision models suffer in a variety of settings that humans capably handle. In light of these phenomena, here we introduce another, orthogonal perspective studying the human-machine vision gap. We revisit the task of recovering images under degradation, first introduced over 30 years ago in the Recognition-by-Components theory of human vision. Specifically, we study the performance and behavior of neural networks on the seemingly simple task of classifying regular polygons at varying orders of degradation along their perimeters. To this end, we implement the Automated Shape Recoverability Test for rapidly generating large-scale datasets of perimeter-degraded regular polygons, modernizing the historically manual creation of image recoverability experiments. We then investigate the capacity of neural networks to recognize and recover such degraded shapes when initialized with different priors. Ultimately, we find that neural networks' behavior on this simple task conflicts with human behavior, raising a fundamental question of the robustness and learning capabilities of modern computer vision models.



## **48. Open Set Relation Extraction via Unknown-Aware Training**

cs.CL

Accepted by ACL2023

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04950v1) [paper-pdf](http://arxiv.org/pdf/2306.04950v1)

**Authors**: Jun Zhao, Xin Zhao, Wenyu Zhan, Qi Zhang, Tao Gui, Zhongyu Wei, Yunwen Chen, Xiang Gao, Xuanjing Huang

**Abstract**: The existing supervised relation extraction methods have achieved impressive performance in a closed-set setting, where the relations during both training and testing remain the same. In a more realistic open-set setting, unknown relations may appear in the test set. Due to the lack of supervision signals from unknown relations, a well-performing closed-set relation extractor can still confidently misclassify them into known relations. In this paper, we propose an unknown-aware training method, regularizing the model by dynamically synthesizing negative instances. To facilitate a compact decision boundary, ``difficult'' negative instances are necessary. Inspired by text adversarial attacks, we adaptively apply small but critical perturbations to original training instances and thus synthesizing negative instances that are more likely to be mistaken by the model as known relations. Experimental results show that this method achieves SOTA unknown relation detection without compromising the classification of known relations.



## **49. Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework**

cs.CL

Accepted to Findings of ACL 2023. Codes are available at:  https://github.com/Phantivia/T-PGD

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2110.15317v4) [paper-pdf](http://arxiv.org/pdf/2110.15317v4)

**Authors**: Lifan Yuan, Yichi Zhang, Yangyi Chen, Wei Wei

**Abstract**: Despite recent success on various tasks, deep learning techniques still perform poorly on adversarial examples with small perturbations. While optimization-based methods for adversarial attacks are well-explored in the field of computer vision, it is impractical to directly apply them in natural language processing due to the discrete nature of the text. To address the problem, we propose a unified framework to extend the existing optimization-based adversarial attack methods in the vision domain to craft textual adversarial samples. In this framework, continuously optimized perturbations are added to the embedding layer and amplified in the forward propagation process. Then the final perturbed latent representations are decoded with a masked language model head to obtain potential adversarial samples. In this paper, we instantiate our framework with an attack algorithm named Textual Projected Gradient Descent (T-PGD). We find our algorithm effective even using proxy gradient information. Therefore, we perform the more challenging transfer black-box attack and conduct comprehensive experiments to evaluate our attack algorithm with several models on three benchmark datasets. Experimental results demonstrate that our method achieves overall better performance and produces more fluent and grammatical adversarial samples compared to strong baseline methods. The code and data are available at \url{https://github.com/Phantivia/T-PGD}.



## **50. Expanding Scope: Adapting English Adversarial Attacks to Chinese**

cs.CL

11 pages; in ACL23 TrustNLP 2023: TrustNLP: Third Workshop on  Trustworthy Natural Language Processing Colocated with the Annual Conference  of the Association for Computational Linguistics (ACL 2023)

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04874v1) [paper-pdf](http://arxiv.org/pdf/2306.04874v1)

**Authors**: Hanyu Liu, Chengyuan Cai, Yanjun Qi

**Abstract**: Recent studies have revealed that NLP predictive models are vulnerable to adversarial attacks. Most existing studies focused on designing attacks to evaluate the robustness of NLP models in the English language alone. Literature has seen an increasing need for NLP solutions for other languages. We, therefore, ask one natural question: whether state-of-the-art (SOTA) attack methods generalize to other languages. This paper investigates how to adapt SOTA adversarial attack algorithms in English to the Chinese language. Our experiments show that attack methods previously applied to English NLP can generate high-quality adversarial examples in Chinese when combined with proper text segmentation and linguistic constraints. In addition, we demonstrate that the generated adversarial examples can achieve high fluency and semantic consistency by focusing on the Chinese language's morphology and phonology, which in turn can be used to improve the adversarial robustness of Chinese NLP models.



