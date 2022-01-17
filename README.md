# Latest Adversarial Attack Papers
**update at 2022-01-17 09:50:01**

[中文版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Unlabeled Data Improves Adversarial Robustness**

stat.ML

Corrected some math typos in the proof of Lemma 1

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/1905.13736v4)

**Authors**: Yair Carmon, Aditi Raghunathan, Ludwig Schmidt, Percy Liang, John C. Duchi

**Abstracts**: We demonstrate, theoretically and empirically, that adversarial robustness can significantly benefit from semisupervised learning. Theoretically, we revisit the simple Gaussian model of Schmidt et al. that shows a sample complexity gap between standard and robust classification. We prove that unlabeled data bridges this gap: a simple semisupervised learning procedure (self-training) achieves high robust accuracy using the same number of labels required for achieving high standard accuracy. Empirically, we augment CIFAR-10 with 500K unlabeled images sourced from 80 Million Tiny Images and use robust self-training to outperform state-of-the-art robust accuracies by over 5 points in (i) $\ell_\infty$ robustness against several strong attacks via adversarial training and (ii) certified $\ell_2$ and $\ell_\infty$ robustness via randomized smoothing. On SVHN, adding the dataset's own extra training set with the labels removed provides gains of 4 to 10 points, within 1 point of the gain from using the extra labels.



## **2. Attention-Guided Black-box Adversarial Attacks with Large-Scale Multiobjective Evolutionary Optimization**

cs.CV

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2101.07512v3)

**Authors**: Jie Wang, Zhaoxia Yin, Jing Jiang, Yang Du

**Abstracts**: Fooling deep neural networks (DNNs) with the black-box optimization has become a popular adversarial attack fashion, as the structural prior knowledge of DNNs is always unknown. Nevertheless, recent black-box adversarial attacks may struggle to balance their attack ability and visual quality of the generated adversarial examples (AEs) in tackling high-resolution images. In this paper, we propose an attention-guided black-box adversarial attack based on the large-scale multiobjective evolutionary optimization, termed as LMOA. By considering the spatial semantic information of images, we firstly take advantage of the attention map to determine the perturbed pixels. Instead of attacking the entire image, reducing the perturbed pixels with the attention mechanism can help to avoid the notorious curse of dimensionality and thereby improves the performance of attacking. Secondly, a large-scale multiobjective evolutionary algorithm is employed to traverse the reduced pixels in the salient region. Benefiting from its characteristics, the generated AEs have the potential to fool target DNNs while being imperceptible by the human vision. Extensive experimental results have verified the effectiveness of the proposed LMOA on the ImageNet dataset. More importantly, it is more competitive to generate high-resolution AEs with better visual quality compared with the existing black-box adversarial attacks.



## **3. On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles**

cs.CV

11 pages, 11 figures

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2201.05057v1)

**Authors**: Qingzhao Zhang, Shengtuo Hu, Jiachen Sun, Qi Alfred Chen, Z. Morley Mao

**Abstracts**: Trajectory prediction is a critical component for autonomous vehicles (AVs) to perform safe planning and navigation. However, few studies have analyzed the adversarial robustness of trajectory prediction or investigated whether the worst-case prediction can still lead to safe planning. To bridge this gap, we study the adversarial robustness of trajectory prediction models by proposing a new adversarial attack that perturbs normal vehicle trajectories to maximize the prediction error. Our experiments on three models and three datasets show that the adversarial prediction increases the prediction error by more than 150%. Our case studies show that if an adversary drives a vehicle close to the target AV following the adversarial trajectory, the AV may make an inaccurate prediction and even make unsafe driving decisions. We also explore possible mitigation techniques via data augmentation and trajectory smoothing.



## **4. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

cs.LG

24 pages, 16 figures, 5 tables

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2111.06628v3)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.



## **5. Evaluation of Four Black-box Adversarial Attacks and Some Query-efficient Improvement Analysis**

cs.CR

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2201.05001v1)

**Authors**: Rui Wang

**Abstracts**: With the fast development of machine learning technologies, deep learning models have been deployed in almost every aspect of everyday life. However, the privacy and security of these models are threatened by adversarial attacks. Among which black-box attack is closer to reality, where limited knowledge can be acquired from the model. In this paper, we provided basic background knowledge about adversarial attack and analyzed four black-box attack algorithms: Bandits, NES, Square Attack and ZOsignSGD comprehensively. We also explored the newly proposed Square Attack method with respect to square size, hoping to improve its query efficiency.



## **6. Captcha Attack: Turning Captchas Against Humanity**

cs.CR

Currently under submission

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2201.04014v2)

**Authors**: Mauro Conti, Luca Pajola, Pier Paolo Tricomi

**Abstracts**: Nowadays, people generate and share massive content on online platforms (e.g., social networks, blogs). In 2021, the 1.9 billion daily active Facebook users posted around 150 thousand photos every minute. Content moderators constantly monitor these online platforms to prevent the spreading of inappropriate content (e.g., hate speech, nudity images). Based on deep learning (DL) advances, Automatic Content Moderators (ACM) help human moderators handle high data volume. Despite their advantages, attackers can exploit weaknesses of DL components (e.g., preprocessing, model) to affect their performance. Therefore, an attacker can leverage such techniques to spread inappropriate content by evading ACM.   In this work, we propose CAPtcha Attack (CAPA), an adversarial technique that allows users to spread inappropriate text online by evading ACM controls. CAPA, by generating custom textual CAPTCHAs, exploits ACM's careless design implementations and internal procedures vulnerabilities. We test our attack on real-world ACM, and the results confirm the ferocity of our simple yet effective attack, reaching up to a 100% evasion success in most cases. At the same time, we demonstrate the difficulties in designing CAPA mitigations, opening new challenges in CAPTCHAs research area.



## **7. Reconstructing Training Data with Informed Adversaries**

cs.CR

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2201.04845v1)

**Authors**: Borja Balle, Giovanni Cherubin, Jamie Hayes

**Abstracts**: Given access to a machine learning model, can an adversary reconstruct the model's training data? This work studies this question from the lens of a powerful informed adversary who knows all the training data points except one. By instantiating concrete attacks, we show it is feasible to reconstruct the remaining data point in this stringent threat model. For convex models (e.g. logistic regression), reconstruction attacks are simple and can be derived in closed-form. For more general models (e.g. neural networks), we propose an attack strategy based on training a reconstructor network that receives as input the weights of the model under attack and produces as output the target data point. We demonstrate the effectiveness of our attack on image classifiers trained on MNIST and CIFAR-10, and systematically investigate which factors of standard machine learning pipelines affect reconstruction success. Finally, we theoretically investigate what amount of differential privacy suffices to mitigate reconstruction attacks by informed adversaries. Our work provides an effective reconstruction attack that model developers can use to assess memorization of individual points in general settings beyond those considered in previous works (e.g. generative language models or access to training gradients); it shows that standard models have the capacity to store enough information to enable high-fidelity reconstruction of training data points; and it demonstrates that differential privacy can successfully mitigate such attacks in a parameter regime where utility degradation is minimal.



## **8. Towards Adversarially Robust Deep Image Denoising**

eess.IV

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2201.04397v2)

**Authors**: Hanshu Yan, Jingfeng Zhang, Jiashi Feng, Masashi Sugiyama, Vincent Y. F. Tan

**Abstracts**: This work systematically investigates the adversarial robustness of deep image denoisers (DIDs), i.e, how well DIDs can recover the ground truth from noisy observations degraded by adversarial perturbations. Firstly, to evaluate DIDs' robustness, we propose a novel adversarial attack, namely Observation-based Zero-mean Attack ({\sc ObsAtk}), to craft adversarial zero-mean perturbations on given noisy images. We find that existing DIDs are vulnerable to the adversarial noise generated by {\sc ObsAtk}. Secondly, to robustify DIDs, we propose an adversarial training strategy, hybrid adversarial training ({\sc HAT}), that jointly trains DIDs with adversarial and non-adversarial noisy data to ensure that the reconstruction quality is high and the denoisers around non-adversarial data are locally smooth. The resultant DIDs can effectively remove various types of synthetic and adversarial noise. We also uncover that the robustness of DIDs benefits their generalization capability on unseen real-world noise. Indeed, {\sc HAT}-trained DIDs can recover high-quality clean images from real-world noise even without training on real noisy data. Extensive experiments on benchmark datasets, including Set68, PolyU, and SIDD, corroborate the effectiveness of {\sc ObsAtk} and {\sc HAT}.



## **9. Security for Machine Learning-based Software Systems: a survey of threats, practices and challenges**

cs.CR

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2201.04736v1)

**Authors**: Huaming Chen, M. Ali Babar

**Abstracts**: The rapid development of Machine Learning (ML) has demonstrated superior performance in many areas, such as computer vision, video and speech recognition. It has now been increasingly leveraged in software systems to automate the core tasks. However, how to securely develop the machine learning-based modern software systems (MLBSS) remains a big challenge, for which the insufficient consideration will largely limit its application in safety-critical domains. One concern is that the present MLBSS development tends to be rush, and the latent vulnerabilities and privacy issues exposed to external users and attackers will be largely neglected and hard to be identified. Additionally, machine learning-based software systems exhibit different liabilities towards novel vulnerabilities at different development stages from requirement analysis to system maintenance, due to its inherent limitations from the model and data and the external adversary capabilities. In this work, we consider that security for machine learning-based software systems may arise by inherent system defects or external adversarial attacks, and the secure development practices should be taken throughout the whole lifecycle. While machine learning has become a new threat domain for existing software engineering practices, there is no such review work covering the topic. Overall, we present a holistic review regarding the security for MLBSS, which covers a systematic understanding from a structure review of three distinct aspects in terms of security threats. Moreover, it provides a thorough state-of-the-practice for MLBSS secure development. Finally, we summarise the literature for system security assurance, and motivate the future research directions with open challenges. We anticipate this work provides sufficient discussion and novel insights to incorporate system security engineering for future exploration.



## **10. Adversarially Robust Classification by Conditional Generative Model Inversion**

cs.LG

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2201.04733v1)

**Authors**: Mitra Alirezaei, Tolga Tasdizen

**Abstracts**: Most adversarial attack defense methods rely on obfuscating gradients. These methods are successful in defending against gradient-based attacks; however, they are easily circumvented by attacks which either do not use the gradient or by attacks which approximate and use the corrected gradient. Defenses that do not obfuscate gradients such as adversarial training exist, but these approaches generally make assumptions about the attack such as its magnitude. We propose a classification model that does not obfuscate gradients and is robust by construction without assuming prior knowledge about the attack. Our method casts classification as an optimization problem where we "invert" a conditional generator trained on unperturbed, natural images to find the class that generates the closest sample to the query image. We hypothesize that a potential source of brittleness against adversarial attacks is the high-to-low-dimensional nature of feed-forward classifiers which allows an adversary to find small perturbations in the input space that lead to large changes in the output space. On the other hand, a generative model is typically a low-to-high-dimensional mapping. While the method is related to Defense-GAN, the use of a conditional generative model and inversion in our model instead of the feed-forward classifier is a critical difference. Unlike Defense-GAN, which was shown to generate obfuscated gradients that are easily circumvented, we show that our method does not obfuscate gradients. We demonstrate that our model is extremely robust against black-box attacks and has improved robustness against white-box attacks compared to naturally trained, feed-forward classifiers.



## **11. Get your Foes Fooled: Proximal Gradient Split Learning for Defense against Model Inversion Attacks on IoMT data**

cs.CR

9 pages, 5 figures, 2 tables

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2201.04569v1)

**Authors**: Sunder Ali Khowaja, Ik Hyun Lee, Kapal Dev, Muhammad Aslam Jarwar, Nawab Muhammad Faseeh Qureshi

**Abstracts**: The past decade has seen a rapid adoption of Artificial Intelligence (AI), specifically the deep learning networks, in Internet of Medical Things (IoMT) ecosystem. However, it has been shown recently that the deep learning networks can be exploited by adversarial attacks that not only make IoMT vulnerable to the data theft but also to the manipulation of medical diagnosis. The existing studies consider adding noise to the raw IoMT data or model parameters which not only reduces the overall performance concerning medical inferences but also is ineffective to the likes of deep leakage from gradients method. In this work, we propose proximal gradient split learning (PSGL) method for defense against the model inversion attacks. The proposed method intentionally attacks the IoMT data when undergoing the deep neural network training process at client side. We propose the use of proximal gradient method to recover gradient maps and a decision-level fusion strategy to improve the recognition performance. Extensive analysis show that the PGSL not only provides effective defense mechanism against the model inversion attacks but also helps in improving the recognition performance on publicly available datasets. We report 17.9$\%$ and 36.9$\%$ gains in accuracy over reconstructed and adversarial attacked images, respectively.



## **12. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

cs.LG

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2106.05087v3)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named "actor" and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.



## **13. Complete Traceability Multimedia Fingerprinting Codes Resistant to Averaging Attack and Adversarial Noise with Optimal Rate**

cs.IT

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2108.09015v3)

**Authors**: Ilya Vorobyev

**Abstracts**: In this paper we consider complete traceability multimedia fingerprinting codes resistant to averaging attacks and adversarial noise. Recently it was shown that there are no such codes for the case of an arbitrary linear attack. However, for the case of averaging attacks complete traceability multimedia fingerprinting codes of exponential cardinality resistant to constant adversarial noise were constructed in 2020 by Egorova et al. We continue this work and provide an improved lower bound on the rate of these codes.



## **14. Game Theory for Adversarial Attacks and Defenses**

cs.LG

With the agreement of my coauthors, I would like to withdraw the  manuscript "Game Theory for Adversarial Attacks and Defenses". Some  experimental procedures were not included in the manuscript, which makes a  part of important claims not meaningful

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2110.06166v3)

**Authors**: Shorya Sharma

**Abstracts**: Adversarial attacks can generate adversarial inputs by applying small but intentionally worst-case perturbations to samples from the dataset, which leads to even state-of-the-art deep neural networks outputting incorrect answers with high confidence. Hence, some adversarial defense techniques are developed to improve the security and robustness of the models and avoid them being attacked. Gradually, a game-like competition between attackers and defenders formed, in which both players would attempt to play their best strategies against each other while maximizing their own payoffs. To solve the game, each player would choose an optimal strategy against the opponent based on the prediction of the opponent's strategy choice. In this work, we are on the defensive side to apply game-theoretic approaches on defending against attacks. We use two randomization methods, random initialization and stochastic activation pruning, to create diversity of networks. Furthermore, we use one denoising technique, super resolution, to improve models' robustness by preprocessing images before attacks. Our experimental results indicate that those three methods can effectively improve the robustness of deep-learning neural networks.



## **15. Similarity-based Gray-box Adversarial Attack Against Deep Face Recognition**

cs.CV

ACCEPTED in IEEE International Conference on Automatic Face and  Gesture Recognition (FG 2021)

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2201.04011v2)

**Authors**: Hanrui Wang, Shuo Wang, Zhe Jin, Yandan Wang, Cunjian Chen, Massimo Tistarell

**Abstracts**: The majority of adversarial attack techniques perform well against deep face recognition when the full knowledge of the system is revealed (\emph{white-box}). However, such techniques act unsuccessfully in the gray-box setting where the face templates are unknown to the attackers. In this work, we propose a similarity-based gray-box adversarial attack (SGADV) technique with a newly developed objective function. SGADV utilizes the dissimilarity score to produce the optimized adversarial example, i.e., similarity-based adversarial attack. This technique applies to both white-box and gray-box attacks against authentication systems that determine genuine or imposter users using the dissimilarity score. To validate the effectiveness of SGADV, we conduct extensive experiments on face datasets of LFW, CelebA, and CelebA-HQ against deep face recognition models of FaceNet and InsightFace in both white-box and gray-box settings. The results suggest that the proposed method significantly outperforms the existing adversarial attack techniques in the gray-box setting. We hence summarize that the similarity-base approaches to develop the adversarial example could satisfactorily cater to the gray-box attack scenarios for de-authentication.



## **16. Tor circuit fingerprinting defenses using adaptive padding**

cs.CR

17 pages

**SubmitDate**: 2022-01-11    [paper-pdf](http://arxiv.org/pdf/2103.03831v2)

**Authors**: George Kadianakis, Theodoros Polyzos, Mike Perry, Kostas Chatzikokolakis

**Abstracts**: Online anonymity and privacy has been based on confusing the adversary by creating indistinguishable network elements. Tor is the largest and most widely deployed anonymity system, designed against realistic modern adversaries. Recently, researchers have managed to fingerprint Tor's circuits -- and hence the type of underlying traffic -- simply by capturing and analyzing traffic traces. In this work, we study the circuit fingerprinting problem, isolating it from website fingerprinting, and revisit previous findings in this model, showing that accurate attacks are possible even when the application-layer traffic is identical. We then proceed to incrementally create defenses against circuit fingerprinting, using a generic adaptive padding framework for Tor based on WTF-PAD. We present a simple defense which delays a fraction of the traffic, as well as a more advanced one which can effectively hide onion service circuits with zero delays. We thoroughly evaluate both defenses, both analytically and experimentally, discovering new subtle fingerprints, but also showing the effectiveness of our defenses.



## **17. An Empirical Assessment of Endpoint Security Systems Against Advanced Persistent Threats Attack Vectors**

cs.CR

This is the revised (and final) version of  https://doi.org/10.3390/jcp1030021 with more EDRs and proper classification  of products into EDRs and EPPs

**SubmitDate**: 2022-01-11    [paper-pdf](http://arxiv.org/pdf/2108.10422v2)

**Authors**: George Karantzas, Constantinos Patsakis

**Abstracts**: Advanced persistent threats pose a significant challenge for blue teams as they apply various attacks over prolonged periods, impeding event correlation and their detection. In this work, we leverage various diverse attack scenarios to assess the efficacy of EDRs and other endpoint security solutions against detecting and preventing APTs. Our results indicate that there is still a lot of room for improvement as state of the art endpoint security systems fail to prevent and log the bulk of the attacks that are reported in this work. Additionally, we discuss methods to tamper with the telemetry providers of EDRs, allowing an adversary to perform a more stealth attack.



## **18. Quantifying Robustness to Adversarial Word Substitutions**

cs.CL

**SubmitDate**: 2022-01-11    [paper-pdf](http://arxiv.org/pdf/2201.03829v1)

**Authors**: Yuting Yang, Pei Huang, FeiFei Ma, Juan Cao, Meishan Zhang, Jian Zhang, Jintao Li

**Abstracts**: Deep-learning-based NLP models are found to be vulnerable to word substitution perturbations. Before they are widely adopted, the fundamental issues of robustness need to be addressed. Along this line, we propose a formal framework to evaluate word-level robustness. First, to study safe regions for a model, we introduce robustness radius which is the boundary where the model can resist any perturbation. As calculating the maximum robustness radius is computationally hard, we estimate its upper and lower bound. We repurpose attack methods as ways of seeking upper bound and design a pseudo-dynamic programming algorithm for a tighter upper bound. Then verification method is utilized for a lower bound. Further, for evaluating the robustness of regions outside a safe radius, we reexamine robustness from another view: quantification. A robustness metric with a rigorous statistical guarantee is introduced to measure the quantification of adversarial examples, which indicates the model's susceptibility to perturbations outside the safe radius. The metric helps us figure out why state-of-the-art models like BERT can be easily fooled by a few word substitutions, but generalize well in the presence of real-world noises.



## **19. Attacking Video Recognition Models with Bullet-Screen Comments**

cs.CV

**SubmitDate**: 2022-01-11    [paper-pdf](http://arxiv.org/pdf/2110.15629v2)

**Authors**: Kai Chen, Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Recent research has demonstrated that Deep Neural Networks (DNNs) are vulnerable to adversarial patches which introduce perceptible but localized changes to the input. Nevertheless, existing approaches have focused on generating adversarial patches on images, their counterparts in videos have been less explored. Compared with images, attacking videos is much more challenging as it needs to consider not only spatial cues but also temporal cues. To close this gap, we introduce a novel adversarial attack in this paper, the bullet-screen comment (BSC) attack, which attacks video recognition models with BSCs. Specifically, adversarial BSCs are generated with a Reinforcement Learning (RL) framework, where the environment is set as the target model and the agent plays the role of selecting the position and transparency of each BSC. By continuously querying the target models and receiving feedback, the agent gradually adjusts its selection strategies in order to achieve a high fooling rate with non-overlapping BSCs. As BSCs can be regarded as a kind of meaningful patch, adding it to a clean video will not affect people' s understanding of the video content, nor will arouse people' s suspicion. We conduct extensive experiments to verify the effectiveness of the proposed method. On both UCF-101 and HMDB-51 datasets, our BSC attack method can achieve about 90\% fooling rate when attacking three mainstream video recognition models, while only occluding \textless 8\% areas in the video. Our code is available at https://github.com/kay-ck/BSC-attack.



## **20. Sequential Randomized Smoothing for Adversarially Robust Speech Recognition**

cs.CL

This update adds some relevant references to past and concurrent work

**SubmitDate**: 2022-01-10    [paper-pdf](http://arxiv.org/pdf/2112.03000v2)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: While Automatic Speech Recognition has been shown to be vulnerable to adversarial attacks, defenses against these attacks are still lagging. Existing, naive defenses can be partially broken with an adaptive attack. In classification tasks, the Randomized Smoothing paradigm has been shown to be effective at defending models. However, it is difficult to apply this paradigm to ASR tasks, due to their complexity and the sequential nature of their outputs. Our paper overcomes some of these challenges by leveraging speech-specific tools like enhancement and ROVER voting to design an ASR model that is robust to perturbations. We apply adaptive versions of state-of-the-art attacks, such as the Imperceptible ASR attack, to our model, and show that our strongest defense is robust to all attacks that use inaudible noise, and can only be broken with very high distortion.



## **21. GMFIM: A Generative Mask-guided Facial Image Manipulation Model for Privacy Preservation**

cs.CV

**SubmitDate**: 2022-01-10    [paper-pdf](http://arxiv.org/pdf/2201.03353v1)

**Authors**: Mohammad Hossein Khojaste, Nastaran Moradzadeh Farid, Ahmad Nickabadi

**Abstracts**: The use of social media websites and applications has become very popular and people share their photos on these networks. Automatic recognition and tagging of people's photos on these networks has raised privacy preservation issues and users seek methods for hiding their identities from these algorithms. Generative adversarial networks (GANs) are shown to be very powerful in generating face images in high diversity and also in editing face images. In this paper, we propose a Generative Mask-guided Face Image Manipulation (GMFIM) model based on GANs to apply imperceptible editing to the input face image to preserve the privacy of the person in the image. Our model consists of three main components: a) the face mask module to cut the face area out of the input image and omit the background, b) the GAN-based optimization module for manipulating the face image and hiding the identity and, c) the merge module for combining the background of the input image and the manipulated de-identified face image. Different criteria are considered in the loss function of the optimization step to produce high-quality images that are as similar as possible to the input image while they cannot be recognized by AFR systems. The results of the experiments on different datasets show that our model can achieve better performance against automated face recognition systems in comparison to the state-of-the-art methods and it catches a higher attack success rate in most experiments from a total of 18. Moreover, the generated images of our proposed model have the highest quality and are more pleasing to human eyes.



## **22. Evaluation of Neural Networks Defenses and Attacks using NDCG and Reciprocal Rank Metrics**

cs.CR

12 pages, 5 figures

**SubmitDate**: 2022-01-10    [paper-pdf](http://arxiv.org/pdf/2201.05071v1)

**Authors**: Haya Brama, Lihi Dery, Tal Grinshpoun

**Abstracts**: The problem of attacks on neural networks through input modification (i.e., adversarial examples) has attracted much attention recently. Being relatively easy to generate and hard to detect, these attacks pose a security breach that many suggested defenses try to mitigate. However, the evaluation of the effect of attacks and defenses commonly relies on traditional classification metrics, without adequate adaptation to adversarial scenarios. Most of these metrics are accuracy-based, and therefore may have a limited scope and low distinctive power. Other metrics do not consider the unique characteristics of neural networks functionality, or measure the effect of the attacks indirectly (e.g., through the complexity of their generation). In this paper, we present two metrics which are specifically designed to measure the effect of attacks, or the recovery effect of defenses, on the output of neural networks in multiclass classification tasks. Inspired by the normalized discounted cumulative gain and the reciprocal rank metrics used in information retrieval literature, we treat the neural network predictions as ranked lists of results. Using additional information about the probability of the rank enabled us to define novel metrics that are suited to the task at hand. We evaluate our metrics using various attacks and defenses on a pretrained VGG19 model and the ImageNet dataset. Compared to the common classification metrics, our proposed metrics demonstrate superior informativeness and distinctiveness.



## **23. IoTGAN: GAN Powered Camouflage Against Machine Learning Based IoT Device Identification**

cs.CR

**SubmitDate**: 2022-01-10    [paper-pdf](http://arxiv.org/pdf/2201.03281v1)

**Authors**: Tao Hou, Tao Wang, Zhuo Lu, Yao Liu, Yalin Sagduyu

**Abstracts**: With the proliferation of IoT devices, researchers have developed a variety of IoT device identification methods with the assistance of machine learning. Nevertheless, the security of these identification methods mostly depends on collected training data. In this research, we propose a novel attack strategy named IoTGAN to manipulate an IoT device's traffic such that it can evade machine learning based IoT device identification. In the development of IoTGAN, we have two major technical challenges: (i) How to obtain the discriminative model in a black-box setting, and (ii) How to add perturbations to IoT traffic through the manipulative model, so as to evade the identification while not influencing the functionality of IoT devices. To address these challenges, a neural network based substitute model is used to fit the target model in black-box settings, it works as a discriminative model in IoTGAN. A manipulative model is trained to add adversarial perturbations into the IoT device's traffic to evade the substitute model. Experimental results show that IoTGAN can successfully achieve the attack goals. We also develop efficient countermeasures to protect machine learning based IoT device identification from been undermined by IoTGAN.



## **24. Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models**

cs.CL

Oral Presentation in NeurIPS 2021 (Datasets and Benchmarks Track). 24  pages, 4 figures, 12 tables

**SubmitDate**: 2022-01-10    [paper-pdf](http://arxiv.org/pdf/2111.02840v2)

**Authors**: Boxin Wang, Chejian Xu, Shuohang Wang, Zhe Gan, Yu Cheng, Jianfeng Gao, Ahmed Hassan Awadallah, Bo Li

**Abstracts**: Large-scale pre-trained language models have achieved tremendous success across a wide range of natural language understanding (NLU) tasks, even surpassing human performance. However, recent studies reveal that the robustness of these models can be challenged by carefully crafted textual adversarial examples. While several individual datasets have been proposed to evaluate model robustness, a principled and comprehensive benchmark is still missing. In this paper, we present Adversarial GLUE (AdvGLUE), a new multi-task benchmark to quantitatively and thoroughly explore and evaluate the vulnerabilities of modern large-scale language models under various types of adversarial attacks. In particular, we systematically apply 14 textual adversarial attack methods to GLUE tasks to construct AdvGLUE, which is further validated by humans for reliable annotations. Our findings are summarized as follows. (i) Most existing adversarial attack algorithms are prone to generating invalid or ambiguous adversarial examples, with around 90% of them either changing the original semantic meanings or misleading human annotators as well. Therefore, we perform a careful filtering process to curate a high-quality benchmark. (ii) All the language models and robust training methods we tested perform poorly on AdvGLUE, with scores lagging far behind the benign accuracy. We hope our work will motivate the development of new adversarial attacks that are more stealthy and semantic-preserving, as well as new robust language models against sophisticated adversarial attacks. AdvGLUE is available at https://adversarialglue.github.io.



## **25. FLAME: Taming Backdoors in Federated Learning**

cs.CR

To appear in the 31st USENIX Security Symposium, August 2022, Boston,  MA, USA

**SubmitDate**: 2022-01-09    [paper-pdf](http://arxiv.org/pdf/2101.02281v3)

**Authors**: Thien Duc Nguyen, Phillip Rieger, Huili Chen, Hossein Yalame, Helen Möllering, Hossein Fereidooni, Samuel Marchal, Markus Miettinen, Azalia Mirhoseini, Shaza Zeitouni, Farinaz Koushanfar, Ahmad-Reza Sadeghi, Thomas Schneider

**Abstracts**: Federated Learning (FL) is a collaborative machine learning approach allowing participants to jointly train a model without having to share their private, potentially sensitive local datasets with others. Despite its benefits, FL is vulnerable to so-called backdoor attacks, in which an adversary injects manipulated model updates into the federated model aggregation process so that the resulting model will provide targeted false predictions for specific adversary-chosen inputs. Proposed defenses against backdoor attacks based on detecting and filtering out malicious model updates consider only very specific and limited attacker models, whereas defenses based on differential privacy-inspired noise injection significantly deteriorate the benign performance of the aggregated model. To address these deficiencies, we introduce FLAME, a defense framework that estimates the sufficient amount of noise to be injected to ensure the elimination of backdoors. To minimize the required amount of noise, FLAME uses a model clustering and weight clipping approach. This ensures that FLAME can maintain the benign performance of the aggregated model while effectively eliminating adversarial backdoors. Our evaluation of FLAME on several datasets stemming from application areas including image classification, word prediction, and IoT intrusion detection demonstrates that FLAME removes backdoors effectively with a negligible impact on the benign performance of the models.



## **26. Privacy-aware Early Detection of COVID-19 through Adversarial Training**

cs.LG

**SubmitDate**: 2022-01-09    [paper-pdf](http://arxiv.org/pdf/2201.03004v1)

**Authors**: Omid Rohanian, Samaneh Kouchaki, Andrew Soltan, Jenny Yang, Morteza Rohanian, Yang Yang, David Clifton

**Abstracts**: Early detection of COVID-19 is an ongoing area of research that can help with triage, monitoring and general health assessment of potential patients and may reduce operational strain on hospitals that cope with the coronavirus pandemic. Different machine learning techniques have been used in the literature to detect coronavirus using routine clinical data (blood tests, and vital signs). Data breaches and information leakage when using these models can bring reputational damage and cause legal issues for hospitals. In spite of this, protecting healthcare models against leakage of potentially sensitive information is an understudied research area. In this work, we examine two machine learning approaches, intended to predict a patient's COVID-19 status using routinely collected and readily available clinical data. We employ adversarial training to explore robust deep learning architectures that protect attributes related to demographic information about the patients. The two models we examine in this work are intended to preserve sensitive information against adversarial attacks and information leakage. In a series of experiments using datasets from the Oxford University Hospitals, Bedfordshire Hospitals NHS Foundation Trust, University Hospitals Birmingham NHS Foundation Trust, and Portsmouth Hospitals University NHS Trust we train and test two neural networks that predict PCR test results using information from basic laboratory blood tests, and vital signs performed on a patients' arrival to hospital. We assess the level of privacy each one of the models can provide and show the efficacy and robustness of our proposed architectures against a comparable baseline. One of our main contributions is that we specifically target the development of effective COVID-19 detection models with built-in mechanisms in order to selectively protect sensitive attributes against adversarial attacks.



## **27. Tiny Adversarial Mulit-Objective Oneshot Neural Architecture Search**

cs.LG

**SubmitDate**: 2022-01-09    [paper-pdf](http://arxiv.org/pdf/2103.00363v2)

**Authors**: Guoyang Xie, Jinbao Wang, Guo Yu, Feng Zheng, Yaochu Jin

**Abstracts**: Due to limited computational cost and energy consumption, most neural network models deployed in mobile devices are tiny. However, tiny neural networks are commonly very vulnerable to attacks. Current research has proved that larger model size can improve robustness, but little research focuses on how to enhance the robustness of tiny neural networks. Our work focuses on how to improve the robustness of tiny neural networks without seriously deteriorating of clean accuracy under mobile-level resources. To this end, we propose a multi-objective oneshot network architecture search (NAS) algorithm to obtain the best trade-off networks in terms of the adversarial accuracy, the clean accuracy and the model size. Specifically, we design a novel search space based on new tiny blocks and channels to balance model size and adversarial performance. Moreover, since the supernet significantly affects the performance of subnets in our NAS algorithm, we reveal the insights into how the supernet helps to obtain the best subnet under white-box adversarial attacks. Concretely, we explore a new adversarial training paradigm by analyzing the adversarial transferability, the width of the supernet and the difference between training the subnets from scratch and fine-tuning. Finally, we make a statistical analysis for the layer-wise combination of certain blocks and channels on the first non-dominated front, which can serve as a guideline to design tiny neural network architectures for the resilience of adversarial perturbations.



## **28. Attacking Vertical Collaborative Learning System Using Adversarial Dominating Inputs**

cs.CR

**SubmitDate**: 2022-01-08    [paper-pdf](http://arxiv.org/pdf/2201.02775v1)

**Authors**: Qi Pang, Yuanyuan Yuan, Shuai Wang

**Abstracts**: Vertical collaborative learning system also known as vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-preserving manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individual organizations.   Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in collaborative learning scenarios.   We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods to synthesize ADIs of various formats and exploit common VFL systems. We further launch greybox fuzz testing, guided by the resiliency score of "victim" participants, to perturb adversary-controlled inputs and systematically explore the VFL attack surface in a privacy-preserving manner. We conduct an in-depth study on the influence of critical parameters and settings in synthesizing ADIs. Our study reveals new VFL attack opportunities, promoting the identification of unknown threats before breaches and building more secure VFL systems.



## **29. Trade-offs between membership privacy & adversarially robust learning**

cs.LG

**SubmitDate**: 2022-01-08    [paper-pdf](http://arxiv.org/pdf/2006.04622v2)

**Authors**: Jamie Hayes

**Abstracts**: Historically, machine learning methods have not been designed with security in mind. In turn, this has given rise to adversarial examples, carefully perturbed input samples aimed to mislead detection at test time, which have been applied to attack spam and malware classification, and more recently to attack image classification. Consequently, an abundance of research has been devoted to designing machine learning methods that are robust to adversarial examples. Unfortunately, there are desiderata besides robustness that a secure and safe machine learning model must satisfy, such as fairness and privacy. Recent work by Song et al. (2019) has shown, empirically, that there exists a trade-off between robust and private machine learning models. Models designed to be robust to adversarial examples often overfit on training data to a larger extent than standard (non-robust) models. If a dataset contains private information, then any statistical test that separates training and test data by observing a model's outputs can represent a privacy breach, and if a model overfits on training data, these statistical tests become easier.   In this work, we identify settings where standard models will overfit to a larger extent in comparison to robust models, and as empirically observed in previous works, settings where the opposite behavior occurs. Thus, it is not necessarily the case that privacy must be sacrificed to achieve robustness. The degree of overfitting naturally depends on the amount of data available for training. We go on to characterize how the training set size factors into the privacy risks exposed by training a robust model on a simple Gaussian data task, and show empirically that our findings hold on image classification benchmark datasets, such as CIFAR-10 and CIFAR-100.



## **30. Detecting CAN Masquerade Attacks with Signal Clustering Similarity**

cs.CR

7 pages, 7 figures, 1 table

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2201.02665v1)

**Authors**: Pablo Moriano, Robert A. Bridges, Michael D. Iannacone

**Abstracts**: Vehicular Controller Area Networks (CANs) are susceptible to cyber attacks of different levels of sophistication. Fabrication attacks are the easiest to administer -- an adversary simply sends (extra) frames on a CAN -- but also the easiest to detect because they disrupt frame frequency. To overcome time-based detection methods, adversaries must administer masquerade attacks by sending frames in lieu of (and therefore at the expected time of) benign frames but with malicious payloads. Research efforts have proven that CAN attacks, and masquerade attacks in particular, can affect vehicle functionality. Examples include causing unintended acceleration, deactivation of vehicle's brakes, as well as steering the vehicle. We hypothesize that masquerade attacks modify the nuanced correlations of CAN signal time series and how they cluster together. Therefore, changes in cluster assignments should indicate anomalous behavior. We confirm this hypothesis by leveraging our previously developed capability for reverse engineering CAN signals (i.e., CAN-D [Controller Area Network Decoder]) and focus on advancing the state of the art for detecting masquerade attacks by analyzing time series extracted from raw CAN frames. Specifically, we demonstrate that masquerade attacks can be detected by computing time series clustering similarity using hierarchical clustering on the vehicle's CAN signals (time series) and comparing the clustering similarity across CAN captures with and without attacks. We test our approach in a previously collected CAN dataset with masquerade attacks (i.e., the ROAD dataset) and develop a forensic tool as a proof of concept to demonstrate the potential of the proposed approach for detecting CAN masquerade attacks.



## **31. Exploring Adversarial Robustness of Multi-Sensor Perception Systems in Self Driving**

cs.CV

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2101.06784v3)

**Authors**: James Tu, Huichen Li, Xinchen Yan, Mengye Ren, Yun Chen, Ming Liang, Eilyan Bitar, Ersin Yumer, Raquel Urtasun

**Abstracts**: Modern self-driving perception systems have been shown to improve upon processing complementary inputs such as LiDAR with images. In isolation, 2D images have been found to be extremely vulnerable to adversarial attacks. Yet, there have been limited studies on the adversarial robustness of multi-modal models that fuse LiDAR features with image features. Furthermore, existing works do not consider physically realizable perturbations that are consistent across the input modalities. In this paper, we showcase practical susceptibilities of multi-sensor detection by placing an adversarial object on top of a host vehicle. We focus on physically realizable and input-agnostic attacks as they are feasible to execute in practice, and show that a single universal adversary can hide different host vehicles from state-of-the-art multi-modal detectors. Our experiments demonstrate that successful attacks are primarily caused by easily corrupted image features. Furthermore, we find that in modern sensor fusion methods which project image features into 3D, adversarial attacks can exploit the projection process to generate false positives across distant regions in 3D. Towards more robust multi-modal perception systems, we show that adversarial training with feature denoising can boost robustness to such attacks significantly. However, we find that standard adversarial defenses still struggle to prevent false positives which are also caused by inaccurate associations between 3D LiDAR points and 2D pixels.



## **32. Adversarial Example Detection for DNN Models: A Review and Experimental Comparison**

cs.CV

Accepted and published in Artificial Intelligence Review journal

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2105.00203v4)

**Authors**: Ahmed Aldahdooh, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Deforges

**Abstracts**: Deep learning (DL) has shown great success in many human-related tasks, which has led to its adoption in many computer vision based applications, such as security surveillance systems, autonomous vehicles and healthcare. Such safety-critical applications have to draw their path to success deployment once they have the capability to overcome safety-critical challenges. Among these challenges are the defense against or/and the detection of the adversarial examples (AEs). Adversaries can carefully craft small, often imperceptible, noise called perturbations to be added to the clean image to generate the AE. The aim of AE is to fool the DL model which makes it a potential risk for DL applications. Many test-time evasion attacks and countermeasures,i.e., defense or detection methods, are proposed in the literature. Moreover, few reviews and surveys were published and theoretically showed the taxonomy of the threats and the countermeasure methods with little focus in AE detection methods. In this paper, we focus on image classification task and attempt to provide a survey for detection methods of test-time evasion attacks on neural network classifiers. A detailed discussion for such methods is provided with experimental results for eight state-of-the-art detectors under different scenarios on four datasets. We also provide potential challenges and future perspectives for this research direction.



## **33. Semantically Stealthy Adversarial Attacks against Segmentation Models**

cs.CV

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2104.01732v3)

**Authors**: Zhenhua Chen, Chuhua Wang, David J. Crandall

**Abstracts**: Segmentation models have been found to be vulnerable to targeted and non-targeted adversarial attacks. However, the resulting segmentation outputs are often so damaged that it is easy to spot an attack. In this paper, we propose semantically stealthy adversarial attacks which can manipulate targeted labels while preserving non-targeted labels at the same time. One challenge is making semantically meaningful manipulations across datasets and models. Another challenge is avoiding damaging non-targeted labels. To solve these challenges, we consider each input image as prior knowledge to generate perturbations. We also design a special regularizer to help extract features. To evaluate our model's performance, we design three basic attack types, namely `vanishing into the context,' `embedding fake labels,' and `displacing target objects.' Our experiments show that our stealthy adversarial model can attack segmentation models with a relatively high success rate on Cityscapes, Mapillary, and BDD100K. Our framework shows good empirical generalization across datasets and models.



## **34. Towards Understanding and Harnessing the Effect of Image Transformation in Adversarial Detection**

cs.CV

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2201.01080v2)

**Authors**: Hui Liu, Bo Zhao, Yuefeng Peng, Weidong Li, Peng Liu

**Abstracts**: Deep neural networks (DNNs) are threatened by adversarial examples. Adversarial detection, which distinguishes adversarial images from benign images, is fundamental for robust DNN-based services. Image transformation is one of the most effective approaches to detect adversarial examples. During the last few years, a variety of image transformations have been studied and discussed to design reliable adversarial detectors. In this paper, we systematically synthesize the recent progress on adversarial detection via image transformations with a novel classification method. Then, we conduct extensive experiments to test the detection performance of image transformations against state-of-the-art adversarial attacks. Furthermore, we reveal that each individual transformation is not capable of detecting adversarial examples in a robust way, and propose a DNN-based approach referred to as AdvJudge, which combines scores of 9 image transformations. Without knowing which individual scores are misleading or not misleading, AdvJudge can make the right judgment, and achieve a significant improvement in detection accuracy. We claim that AdvJudge is a more effective adversarial detector than those based on an individual image transformation.



## **35. BDFA: A Blind Data Adversarial Bit-flip Attack on Deep Neural Networks**

cs.CR

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2112.03477v2)

**Authors**: Behnam Ghavami, Mani Sadati, Mohammad Shahidzadeh, Zhenman Fang, Lesley Shannon

**Abstracts**: Adversarial bit-flip attack (BFA) on Neural Network weights can result in catastrophic accuracy degradation by flipping a very small number of bits. A major drawback of prior bit flip attack techniques is their reliance on test data. This is frequently not possible for applications that contain sensitive or proprietary data. In this paper, we propose Blind Data Adversarial Bit-flip Attack (BDFA), a novel technique to enable BFA without any access to the training or testing data. This is achieved by optimizing for a synthetic dataset, which is engineered to match the statistics of batch normalization across different layers of the network and the targeted label. Experimental results show that BDFA could decrease the accuracy of ResNet50 significantly from 75.96\% to 13.94\% with only 4 bits flips.



## **36. Accelerated Zeroth-Order and First-Order Momentum Methods from Mini to Minimax Optimization**

math.OC

Published in Journal of Machine Learning Research (JMLR)

**SubmitDate**: 2022-01-06    [paper-pdf](http://arxiv.org/pdf/2008.08170v6)

**Authors**: Feihu Huang, Shangqian Gao, Jian Pei, Heng Huang

**Abstracts**: In the paper, we propose a class of accelerated zeroth-order and first-order momentum methods for both nonconvex mini-optimization and minimax-optimization. Specifically, we propose a new accelerated zeroth-order momentum (Acc-ZOM) method for black-box mini-optimization. Moreover, we prove that our Acc-ZOM method achieves a lower query complexity of $\tilde{O}(d^{3/4}\epsilon^{-3})$ for finding an $\epsilon$-stationary point, which improves the best known result by a factor of $O(d^{1/4})$ where $d$ denotes the variable dimension. In particular, the Acc-ZOM does not require large batches required in the existing zeroth-order stochastic algorithms. Meanwhile, we propose an accelerated \textbf{zeroth-order} momentum descent ascent (Acc-ZOMDA) method for \textbf{black-box} minimax-optimization, which obtains a query complexity of $\tilde{O}((d_1+d_2)^{3/4}\kappa_y^{4.5}\epsilon^{-3})$ without large batches for finding an $\epsilon$-stationary point, where $d_1$ and $d_2$ denote variable dimensions and $\kappa_y$ is condition number. Moreover, we propose an accelerated \textbf{first-order} momentum descent ascent (Acc-MDA) method for \textbf{white-box} minimax optimization, which has a gradient complexity of $\tilde{O}(\kappa_y^{4.5}\epsilon^{-3})$ without large batches for finding an $\epsilon$-stationary point. In particular, our Acc-MDA can obtain a lower gradient complexity of $\tilde{O}(\kappa_y^{2.5}\epsilon^{-3})$ with a batch size $O(\kappa_y^4)$. Extensive experimental results on black-box adversarial attack to deep neural networks and poisoning attack to logistic regression demonstrate efficiency of our algorithms.



## **37. Increased-confidence adversarial examples for deep learning counter-forensics**

cs.CV

**SubmitDate**: 2022-01-06    [paper-pdf](http://arxiv.org/pdf/2005.06023v2)

**Authors**: Wenjie Li, Benedetta Tondi, Rongrong Ni, Mauro Barni

**Abstracts**: Transferability of adversarial examples is a key issue to apply this kind of attacks against multimedia forensics (MMF) techniques based on Deep Learning (DL) in a real-life setting. Adversarial example transferability, in fact, would open the way to the deployment of successful counter forensics attacks also in cases where the attacker does not have a full knowledge of the to-be-attacked system. Some preliminary works have shown that adversarial examples against CNN-based image forensics detectors are in general non-transferrable, at least when the basic versions of the attacks implemented in the most popular libraries are adopted. In this paper, we introduce a general strategy to increase the strength of the attacks and evaluate their transferability when such a strength varies. We experimentally show that, in this way, attack transferability can be largely increased, at the expense of a larger distortion. Our research confirms the security threats posed by the existence of adversarial examples even in multimedia forensics scenarios, thus calling for new defense strategies to improve the security of DL-based MMF techniques.



## **38. HydraText: Multi-objective Optimization for Adversarial Textual Attack**

cs.CL

**SubmitDate**: 2022-01-06    [paper-pdf](http://arxiv.org/pdf/2111.01528v2)

**Authors**: Shengcai Liu, Ning Lu, Wenjing Hong, Chao Qian, Ke Tang

**Abstracts**: The field of adversarial textual attack has significantly grown over the last few years, where the commonly considered objective is to craft adversarial examples (AEs) that can successfully fool the target model. However, the imperceptibility of attacks, which is also an essential objective for practical attackers, is often left out by previous studies. In consequence, the crafted AEs tend to have obvious structural and semantic differences from the original human-written texts, making them easily perceptible. In this paper, we advocate simultaneously considering both objectives of successful and imperceptible attacks. Specifically, we formulate the problem of crafting AEs as a multi-objective set maximization problem, and propose a novel evolutionary algorithm (dubbed HydraText) to solve it. To the best of our knowledge, HydraText is currently the only approach that can be effectively applied to both score-based and decision-based attack settings. Exhaustive experiments involving 44237 instances demonstrate that HydraText consistently achieves higher attack success rates and better attack imperceptibility than the state-of-the-art textual attack approaches. A human evaluation study also shows that the AEs crafted by HydraText are more indistinguishable from human-written texts. Finally, these AEs exhibit good transferability and can bring notable robustness improvement to the target models by adversarial training.



## **39. Increasing the Confidence of Deep Neural Networks by Coverage Analysis**

cs.LG

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2101.12100v3)

**Authors**: Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: The great performance of machine learning algorithms and deep neural networks in several perception and control tasks is pushing the industry to adopt such technologies in safety-critical applications, as autonomous robots and self-driving vehicles. At present, however, several issues need to be solved to make deep learning methods more trustworthy, predictable, safe, and secure against adversarial attacks. Although several methods have been proposed to improve the trustworthiness of deep neural networks, most of them are tailored for specific classes of adversarial examples, hence failing to detect other corner cases or unsafe inputs that heavily deviate from the training samples. This paper presents a lightweight monitoring architecture based on coverage paradigms to enhance the model robustness against different unsafe inputs. In particular, four coverage analysis methods are proposed and tested in the architecture for evaluating multiple detection logics. Experimental results show that the proposed approach is effective in detecting both powerful adversarial examples and out-of-distribution inputs, introducing limited extra-execution time and memory requirements.



## **40. On the Real-World Adversarial Robustness of Real-Time Semantic Segmentation Models for Autonomous Driving**

cs.CV

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2201.01850v1)

**Authors**: Giulio Rossolini, Federico Nesti, Gianluca D'Amico, Saasha Nair, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: The existence of real-world adversarial examples (commonly in the form of patches) poses a serious threat for the use of deep learning models in safety-critical computer vision tasks such as visual perception in autonomous driving. This paper presents an extensive evaluation of the robustness of semantic segmentation models when attacked with different types of adversarial patches, including digital, simulated, and physical ones. A novel loss function is proposed to improve the capabilities of attackers in inducing a misclassification of pixels. Also, a novel attack strategy is presented to improve the Expectation Over Transformation method for placing a patch in the scene. Finally, a state-of-the-art method for detecting adversarial patch is first extended to cope with semantic segmentation models, then improved to obtain real-time performance, and eventually evaluated in real-world scenarios. Experimental results reveal that, even though the adversarial effect is visible with both digital and real-world attacks, its impact is often spatially confined to areas of the image around the patch. This opens to further questions about the spatial robustness of real-time semantic segmentation models.



## **41. Secure Remote Attestation with Strong Key Insulation Guarantees**

cs.CR

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2201.01834v1)

**Authors**: Deniz Gurevin, Chenglu Jin, Phuong Ha Nguyen, Omer Khan, Marten van Dijk

**Abstracts**: Recent years have witnessed a trend of secure processor design in both academia and industry. Secure processors with hardware-enforced isolation can be a solid foundation of cloud computation in the future. However, due to recent side-channel attacks, the commercial secure processors failed to deliver the promises of a secure isolated execution environment. Sensitive information inside the secure execution environment always gets leaked via side channels. This work considers the most powerful software-based side-channel attackers, i.e., an All Digital State Observing (ADSO) adversary who can observe all digital states, including all digital states in secure enclaves. Traditional signature schemes are not secure in ADSO adversarial model. We introduce a new cryptographic primitive called One-Time Signature with Secret Key Exposure (OTS-SKE), which ensures no one can forge a valid signature of a new message or nonce even if all secret session keys are leaked. OTS-SKE enables us to sign attestation reports securely under the ADSO adversary. We also minimize the trusted computing base by introducing a secure co-processor into the system, and the interaction between the secure co-processor and the attestation processor is unidirectional. That is, the co-processor takes no inputs from the processor and only generates secret keys for the processor to fetch. Our experimental results show that the signing of OTS-SKE is faster than that of Elliptic Curve Digital Signature Algorithm (ECDSA) used in Intel SGX.



## **42. Generation of Wheel Lockup Attacks on Nonlinear Dynamics of Vehicle Traction**

eess.SY

Submitted to AutoSec'22@NDSS

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2112.09229v2)

**Authors**: Alireza Mohammadi, Hafiz Malik, Masoud Abbaszadeh

**Abstracts**: There is ample evidence in the automotive cybersecurity literature that the car brake ECUs can be maliciously reprogrammed. Motivated by such threat, this paper investigates the capabilities of an adversary who can directly control the frictional brake actuators and would like to induce wheel lockup conditions leading to catastrophic road injuries. This paper demonstrates that the adversary despite having a limited knowledge of the tire-road interaction characteristics has the capability of driving the states of the vehicle traction dynamics to a vicinity of the lockup manifold in a finite time by means of a properly designed attack policy for the frictional brakes. This attack policy relies on employing a predefined-time controller and a nonlinear disturbance observer acting on the wheel slip error dynamics. Simulations under various road conditions demonstrate the effectiveness of the proposed attack policy.



## **43. GRNN: Generative Regression Neural Network -- A Data Leakage Attack for Federated Learning**

cs.LG

The source code can be found at: https://github.com/Rand2AI/GRNN

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2105.00529v2)

**Authors**: Hanchi Ren, Jingjing Deng, Xianghua Xie

**Abstracts**: Data privacy has become an increasingly important issue in Machine Learning (ML), where many approaches have been developed to tackle this challenge, e.g. cryptography (Homomorphic Encryption (HE), Differential Privacy (DP), etc.) and collaborative training (Secure Multi-Party Computation (MPC), Distributed Learning and Federated Learning (FL)). These techniques have a particular focus on data encryption or secure local computation. They transfer the intermediate information to the third party to compute the final result. Gradient exchanging is commonly considered to be a secure way of training a robust model collaboratively in Deep Learning (DL). However, recent researches have demonstrated that sensitive information can be recovered from the shared gradient. Generative Adversarial Network (GAN), in particular, has shown to be effective in recovering such information. However, GAN based techniques require additional information, such as class labels which are generally unavailable for privacy-preserved learning. In this paper, we show that, in the FL system, image-based privacy data can be easily recovered in full from the shared gradient only via our proposed Generative Regression Neural Network (GRNN). We formulate the attack to be a regression problem and optimize two branches of the generative model by minimizing the distance between gradients. We evaluate our method on several image classification tasks. The results illustrate that our proposed GRNN outperforms state-of-the-art methods with better stability, stronger robustness, and higher accuracy. It also has no convergence requirement to the global FL model. Moreover, we demonstrate information leakage using face re-identification. Some defense strategies are also discussed in this work.



## **44. Aspis: A Robust Detection System for Distributed Learning**

cs.LG

17 pages, 23 figures

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2108.02416v2)

**Authors**: Konstantinos Konstantinidis, Aditya Ramamoorthy

**Abstracts**: State-of-the-art machine learning models are routinely trained on large-scale distributed clusters. Crucially, such systems can be compromised when some of the computing devices exhibit abnormal (Byzantine) behavior and return arbitrary results to the parameter server (PS). This behavior may be attributed to a plethora of reasons, including system failures and orchestrated attacks. Existing work suggests robust aggregation and/or computational redundancy to alleviate the effect of distorted gradients. However, most of these schemes are ineffective when an adversary knows the task assignment and can choose the attacked workers judiciously to induce maximal damage. Our proposed method Aspis assigns gradient computations to worker nodes using a subset-based assignment which allows for multiple consistency checks on the behavior of a worker node. Examination of the calculated gradients and post-processing (clique-finding in an appropriately constructed graph) by the central node allows for efficient detection and subsequent exclusion of adversaries from the training process. We prove the Byzantine resilience and detection guarantees of Aspis under weak and strong attacks and extensively evaluate the system on various large-scale training scenarios. The principal metric for our experiments is the test accuracy, for which we demonstrate a significant improvement of about 30% compared to many state-of-the-art approaches on the CIFAR-10 dataset. The corresponding reduction of the fraction of corrupted gradients ranges from 16% to 99%.



## **45. ROOM: Adversarial Machine Learning Attacks Under Real-Time Constraints**

cs.CR

12 pages

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2201.01621v1)

**Authors**: Amira Guesmi, Khaled N. Khasawneh, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstracts**: Advances in deep learning have enabled a wide range of promising applications. However, these systems are vulnerable to Adversarial Machine Learning (AML) attacks; adversarially crafted perturbations to their inputs could cause them to misclassify. Several state-of-the-art adversarial attacks have demonstrated that they can reliably fool classifiers making these attacks a significant threat. Adversarial attack generation algorithms focus primarily on creating successful examples while controlling the noise magnitude and distribution to make detection more difficult. The underlying assumption of these attacks is that the adversarial noise is generated offline, making their execution time a secondary consideration. However, recently, just-in-time adversarial attacks where an attacker opportunistically generates adversarial examples on the fly have been shown to be possible. This paper introduces a new problem: how do we generate adversarial noise under real-time constraints to support such real-time adversarial attacks? Understanding this problem improves our understanding of the threat these attacks pose to real-time systems and provides security evaluation benchmarks for future defenses. Therefore, we first conduct a run-time analysis of adversarial generation algorithms. Universal attacks produce a general attack offline, with no online overhead, and can be applied to any input; however, their success rate is limited because of their generality. In contrast, online algorithms, which work on a specific input, are computationally expensive, making them inappropriate for operation under time constraints. Thus, we propose ROOM, a novel Real-time Online-Offline attack construction Model where an offline component serves to warm up the online algorithm, making it possible to generate highly successful attacks under time constraints.



## **46. A Survey on Adversarial Attacks for Malware Analysis**

cs.CR

48 Pages, 31 Figures, 11 Tables

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2111.08223v2)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam

**Abstracts**: Machine learning has witnessed tremendous growth in its adoption and advancement in the last decade. The evolution of machine learning from traditional algorithms to modern deep learning architectures has shaped the way today's technology functions. Its unprecedented ability to discover knowledge/patterns from unstructured data and automate the decision-making process led to its application in wide domains. High flying machine learning arena has been recently pegged back by the introduction of adversarial attacks. Adversaries are able to modify data, maximizing the classification error of the models. The discovery of blind spots in machine learning models has been exploited by adversarial attackers by generating subtle intentional perturbations in test samples. Increasing dependency on data has paved the blueprint for ever-high incentives to camouflage machine learning models. To cope with probable catastrophic consequences in the future, continuous research is required to find vulnerabilities in form of adversarial and design remedies in systems. This survey aims at providing the encyclopedic introduction to adversarial attacks that are carried out against malware detection systems. The paper will introduce various machine learning techniques used to generate adversarial and explain the structure of target files. The survey will also model the threat posed by the adversary and followed by brief descriptions of widely accepted adversarial algorithms. Work will provide a taxonomy of adversarial evasion attacks on the basis of attack domain and adversarial generation techniques. Adversarial evasion attacks carried out against malware detectors will be discussed briefly under each taxonomical headings and compared with concomitant researches. Analyzing the current research challenges in an adversarial generation, the survey will conclude by pinpointing the open future research directions.



## **47. Fast Gradient Non-sign Methods**

cs.CV

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2110.12734v2)

**Authors**: Yaya Cheng, Xiaosu Zhu, Qilong Zhang, Lianli Gao, Jingkuan Song

**Abstracts**: Adversarial attacks make their success in "fooling" DNNs and among them, gradient-based algorithms become one of the mainstreams. Based on the linearity hypothesis [12], under $\ell_\infty$ constraint, $sign$ operation applied to the gradients is a good choice for generating perturbations. However, the side-effect from such operation exists since it leads to the bias of direction between the real gradients and the perturbations. In other words, current methods contain a gap between real gradients and actual noises, which leads to biased and inefficient attacks. Therefore in this paper, based on the Taylor expansion, the bias is analyzed theoretically and the correction of $\sign$, i.e., Fast Gradient Non-sign Method (FGNM), is further proposed. Notably, FGNM is a general routine, which can seamlessly replace the conventional $sign$ operation in gradient-based attacks with negligible extra computational cost. Extensive experiments demonstrate the effectiveness of our methods. Specifically, ours outperform them by \textbf{27.5\%} at most and \textbf{9.5\%} on average. Our anonymous code is publicly available: \url{https://git.io/mm-fgnm}.



## **48. Adversarial Feature Desensitization**

cs.LG

Accepted at Neurips 2021

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2006.04621v3)

**Authors**: Pouya Bashivan, Reza Bayat, Adam Ibrahim, Kartik Ahuja, Mojtaba Faramarzi, Touraj Laleh, Blake Aaron Richards, Irina Rish

**Abstracts**: Neural networks are known to be vulnerable to adversarial attacks -- slight but carefully constructed perturbations of the inputs which can drastically impair the network's performance. Many defense methods have been proposed for improving robustness of deep networks by training them on adversarially perturbed inputs. However, these models often remain vulnerable to new types of attacks not seen during training, and even to slightly stronger versions of previously seen attacks. In this work, we propose a novel approach to adversarial robustness, which builds upon the insights from the domain adaptation field. Our method, called Adversarial Feature Desensitization (AFD), aims at learning features that are invariant towards adversarial perturbations of the inputs. This is achieved through a game where we learn features that are both predictive and robust (insensitive to adversarial attacks), i.e. cannot be used to discriminate between natural and adversarial data. Empirical results on several benchmarks demonstrate the effectiveness of the proposed approach against a wide range of attack types and attack strengths. Our code is available at https://github.com/BashivanLab/afd.



## **49. On the Minimal Adversarial Perturbation for Deep Neural Networks with Provable Estimation Error**

cs.LG

Under review on IEEE journal Transactions on Pattern Analysis and  Machine Intelligence

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2201.01235v1)

**Authors**: Fabio Brau, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: Although Deep Neural Networks (DNNs) have shown incredible performance in perceptive and control tasks, several trustworthy issues are still open. One of the most discussed topics is the existence of adversarial perturbations, which has opened an interesting research line on provable techniques capable of quantifying the robustness of a given input. In this regard, the Euclidean distance of the input from the classification boundary denotes a well-proved robustness assessment as the minimal affordable adversarial perturbation. Unfortunately, computing such a distance is highly complex due the non-convex nature of NNs. Despite several methods have been proposed to address this issue, to the best of our knowledge, no provable results have been presented to estimate and bound the error committed. This paper addresses this issue by proposing two lightweight strategies to find the minimal adversarial perturbation. Differently from the state-of-the-art, the proposed approach allows formulating an error estimation theory of the approximate distance with respect to the theoretical one. Finally, a substantial set of experiments is reported to evaluate the performance of the algorithms and support the theoretical findings. The obtained results show that the proposed strategies approximate the theoretical distance for samples close to the classification boundary, leading to provable robustness guarantees against any adversarial attacks.



## **50. Adversarial Transformation of Spoofing Attacks for Voice Biometrics**

eess.AS

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2201.01226v1)

**Authors**: Alejandro Gomez-Alanis, Jose A. Gonzalez-Lopez, Antonio M. Peinado

**Abstracts**: Voice biometric systems based on automatic speaker verification (ASV) are exposed to \textit{spoofing} attacks which may compromise their security. To increase the robustness against such attacks, anti-spoofing or presentation attack detection (PAD) systems have been proposed for the detection of replay, synthesis and voice conversion based attacks. Recently, the scientific community has shown that PAD systems are also vulnerable to adversarial attacks. However, to the best of our knowledge, no previous work have studied the robustness of full voice biometrics systems (ASV + PAD) to these new types of adversarial \textit{spoofing} attacks. In this work, we develop a new adversarial biometrics transformation network (ABTN) which jointly processes the loss of the PAD and ASV systems in order to generate white-box and black-box adversarial \textit{spoofing} attacks. The core idea of this system is to generate adversarial \textit{spoofing} attacks which are able to fool the PAD system without being detected by the ASV system. The experiments were carried out on the ASVspoof 2019 corpus, including both logical access (LA) and physical access (PA) scenarios. The experimental results show that the proposed ABTN clearly outperforms some well-known adversarial techniques in both white-box and black-box attack scenarios.



