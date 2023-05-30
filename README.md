# Latest Adversarial Attack Papers
**update at 2023-05-30 09:51:22**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. BITE: Textual Backdoor Attacks with Iterative Trigger Injection**

cs.CL

Accepted to ACL 2023

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2205.12700v3) [paper-pdf](http://arxiv.org/pdf/2205.12700v3)

**Authors**: Jun Yan, Vansh Gupta, Xiang Ren

**Abstract**: Backdoor attacks have become an emerging threat to NLP systems. By providing poisoned training data, the adversary can embed a "backdoor" into the victim model, which allows input instances satisfying certain textual patterns (e.g., containing a keyword) to be predicted as a target label of the adversary's choice. In this paper, we demonstrate that it is possible to design a backdoor attack that is both stealthy (i.e., hard to notice) and effective (i.e., has a high attack success rate). We propose BITE, a backdoor attack that poisons the training data to establish strong correlations between the target label and a set of "trigger words". These trigger words are iteratively identified and injected into the target-label instances through natural word-level perturbations. The poisoned training data instruct the victim model to predict the target label on inputs containing trigger words, forming the backdoor. Experiments on four text classification datasets show that our proposed attack is significantly more effective than baseline methods while maintaining decent stealthiness, raising alarm on the usage of untrusted training data. We further propose a defense method named DeBITE based on potential trigger word removal, which outperforms existing methods in defending against BITE and generalizes well to handling other backdoor attacks.



## **2. Fundamental Limitations of Alignment in Large Language Models**

cs.CL

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2304.11082v2) [paper-pdf](http://arxiv.org/pdf/2304.11082v2)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback increase the LLM's proneness to being prompted into the undesired behaviors. Moreover, we include the notion of personas in our BEB framework, and find that behaviors which are generally very unlikely to be exhibited by the model can be brought to the front by prompting the model to behave as specific persona. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.



## **3. Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition**

cs.CV

17 pages, 13 figures

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.17939v1) [paper-pdf](http://arxiv.org/pdf/2305.17939v1)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstract**: Using Fourier analysis, we explore the robustness and vulnerability of graph convolutional neural networks (GCNs) for skeleton-based action recognition. We adopt a joint Fourier transform (JFT), a combination of the graph Fourier transform (GFT) and the discrete Fourier transform (DFT), to examine the robustness of adversarially-trained GCNs against adversarial attacks and common corruptions. Experimental results with the NTU RGB+D dataset reveal that adversarial training does not introduce a robustness trade-off between adversarial attacks and low-frequency perturbations, which typically occurs during image classification based on convolutional neural networks. This finding indicates that adversarial training is a practical approach to enhancing robustness against adversarial attacks and common corruptions in skeleton-based action recognition. Furthermore, we find that the Fourier approach cannot explain vulnerability against skeletal part occlusion corruption, which highlights its limitations. These findings extend our understanding of the robustness of GCNs, potentially guiding the development of more robust learning methods for skeleton-based action recognition.



## **4. NaturalFinger: Generating Natural Fingerprint with Generative Adversarial Networks**

cs.CV

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.17868v1) [paper-pdf](http://arxiv.org/pdf/2305.17868v1)

**Authors**: Kang Yang, Kunhao Lai

**Abstract**: Deep neural network (DNN) models have become a critical asset of the model owner as training them requires a large amount of resource (i.e. labeled data). Therefore, many fingerprinting schemes have been proposed to safeguard the intellectual property (IP) of the model owner against model extraction and illegal redistribution. However, previous schemes adopt unnatural images as the fingerprint, such as adversarial examples and noisy images, which can be easily perceived and rejected by the adversary. In this paper, we propose NaturalFinger which generates natural fingerprint with generative adversarial networks (GANs). Besides, our proposed NaturalFinger fingerprints the decision difference areas rather than the decision boundary, which is more robust. The application of GAN not only allows us to generate more imperceptible samples, but also enables us to generate unrestricted samples to explore the decision boundary.To demonstrate the effectiveness of our fingerprint approach, we evaluate our approach against four model modification attacks including adversarial training and two model extraction attacks. Experiments show that our approach achieves 0.91 ARUC value on the FingerBench dataset (154 models), exceeding the optimal baseline (MetaV) over 17\%.



## **5. NOTABLE: Transferable Backdoor Attacks Against Prompt-based NLP Models**

cs.CL

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.17826v1) [paper-pdf](http://arxiv.org/pdf/2305.17826v1)

**Authors**: Kai Mei, Zheng Li, Zhenting Wang, Yang Zhang, Shiqing Ma

**Abstract**: Prompt-based learning is vulnerable to backdoor attacks. Existing backdoor attacks against prompt-based models consider injecting backdoors into the entire embedding layers or word embedding vectors. Such attacks can be easily affected by retraining on downstream tasks and with different prompting strategies, limiting the transferability of backdoor attacks. In this work, we propose transferable backdoor attacks against prompt-based models, called NOTABLE, which is independent of downstream tasks and prompting strategies. Specifically, NOTABLE injects backdoors into the encoders of PLMs by utilizing an adaptive verbalizer to bind triggers to specific words (i.e., anchors). It activates the backdoor by pasting input with triggers to reach adversary-desired anchors, achieving independence from downstream tasks and prompting strategies. We conduct experiments on six NLP tasks, three popular models, and three prompting strategies. Empirical results show that NOTABLE achieves superior attack performance (i.e., attack success rate over 90% on all the datasets), and outperforms two state-of-the-art baselines. Evaluations on three defenses show the robustness of NOTABLE. Our code can be found at https://github.com/RU-System-Software-and-Security/Notable.



## **6. DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection**

cs.CV

Code will be available at https://github.com/joellliu/DiffProtect/

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.13625v2) [paper-pdf](http://arxiv.org/pdf/2305.13625v2)

**Authors**: Jiang Liu, Chun Pong Lau, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.



## **7. Amplification trojan network: Attack deep neural networks by amplifying their inherent weakness**

cs.CR

Published Sep 2022 in Neurocomputing

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.17688v1) [paper-pdf](http://arxiv.org/pdf/2305.17688v1)

**Authors**: Zhanhao Hu, Jun Zhu, Bo Zhang, Xiaolin Hu

**Abstract**: Recent works found that deep neural networks (DNNs) can be fooled by adversarial examples, which are crafted by adding adversarial noise on clean inputs. The accuracy of DNNs on adversarial examples will decrease as the magnitude of the adversarial noise increase. In this study, we show that DNNs can be also fooled when the noise is very small under certain circumstances. This new type of attack is called Amplification Trojan Attack (ATAttack). Specifically, we use a trojan network to transform the inputs before sending them to the target DNN. This trojan network serves as an amplifier to amplify the inherent weakness of the target DNN. The target DNN, which is infected by the trojan network, performs normally on clean data while being more vulnerable to adversarial examples. Since it only transforms the inputs, the trojan network can hide in DNN-based pipelines, e.g. by infecting the pre-processing procedure of the inputs before sending them to the DNNs. This new type of threat should be considered in developing safe DNNs.



## **8. Threat Models over Space and Time: A Case Study of E2EE Messaging Applications**

cs.CR

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2301.05653v2) [paper-pdf](http://arxiv.org/pdf/2301.05653v2)

**Authors**: Partha Das Chowdhury, Maria Sameen, Jenny Blessing, Nicholas Boucher, Joseph Gardiner, Tom Burrows, Ross Anderson, Awais Rashid

**Abstract**: Threat modelling is foundational to secure systems engineering and should be done in consideration of the context within which systems operate. On the other hand, the continuous evolution of both the technical sophistication of threats and the system attack surface is an inescapable reality. In this work, we explore the extent to which real-world systems engineering reflects the changing threat context. To this end we examine the desktop clients of six widely used end-to-end-encrypted mobile messaging applications to understand the extent to which they adjusted their threat model over space (when enabling clients on new platforms, such as desktop clients) and time (as new threats emerged). We experimented with short-lived adversarial access against these desktop clients and analyzed the results with respect to two popular threat elicitation frameworks, STRIDE and LINDDUN. The results demonstrate that system designers need to both recognise the threats in the evolving context within which systems operate and, more importantly, to mitigate them by rescoping trust boundaries in a manner that those within the administrative boundary cannot violate security and privacy properties. Such a nuanced understanding of trust boundary scopes and their relationship with administrative boundaries allows for better administration of shared components, including securing them with safe defaults.



## **9. SneakyPrompt: Evaluating Robustness of Text-to-image Generative Models' Safety Filters**

cs.LG

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.12082v2) [paper-pdf](http://arxiv.org/pdf/2305.12082v2)

**Authors**: Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, Yinzhi Cao

**Abstract**: Text-to-image generative models such as Stable Diffusion and DALL$\cdot$E 2 have attracted much attention since their publication due to their wide application in the real world. One challenging problem of text-to-image generative models is the generation of Not-Safe-for-Work (NSFW) content, e.g., those related to violence and adult. Therefore, a common practice is to deploy a so-called safety filter, which blocks NSFW content based on either text or image features. Prior works have studied the possible bypass of such safety filters. However, existing works are largely manual and specific to Stable Diffusion's official safety filter. Moreover, the bypass ratio of Stable Diffusion's safety filter is as low as 23.51% based on our evaluation.   In this paper, we propose the first automated attack framework, called SneakyPrompt, to evaluate the robustness of real-world safety filters in state-of-the-art text-to-image generative models. Our key insight is to search for alternative tokens in a prompt that generates NSFW images so that the generated prompt (called an adversarial prompt) bypasses existing safety filters. Specifically, SneakyPrompt utilizes reinforcement learning (RL) to guide an agent with positive rewards on semantic similarity and bypass success.   Our evaluation shows that SneakyPrompt successfully generated NSFW content using an online model DALL$\cdot$E 2 with its default, closed-box safety filter enabled. At the same time, we also deploy several open-source state-of-the-art safety filters on a Stable Diffusion model and show that SneakyPrompt not only successfully generates NSFW content, but also outperforms existing adversarial attacks in terms of the number of queries and image qualities.



## **10. Tubes Among Us: Analog Attack on Automatic Speaker Identification**

cs.LG

Published at USENIX Security 2023  https://www.usenix.org/conference/usenixsecurity23/presentation/ahmed

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2202.02751v2) [paper-pdf](http://arxiv.org/pdf/2202.02751v2)

**Authors**: Shimaa Ahmed, Yash Wani, Ali Shahin Shamsabadi, Mohammad Yaghini, Ilia Shumailov, Nicolas Papernot, Kassem Fawaz

**Abstract**: Recent years have seen a surge in the popularity of acoustics-enabled personal devices powered by machine learning. Yet, machine learning has proven to be vulnerable to adversarial examples. A large number of modern systems protect themselves against such attacks by targeting artificiality, i.e., they deploy mechanisms to detect the lack of human involvement in generating the adversarial examples. However, these defenses implicitly assume that humans are incapable of producing meaningful and targeted adversarial examples. In this paper, we show that this base assumption is wrong. In particular, we demonstrate that for tasks like speaker identification, a human is capable of producing analog adversarial examples directly with little cost and supervision: by simply speaking through a tube, an adversary reliably impersonates other speakers in eyes of ML models for speaker identification. Our findings extend to a range of other acoustic-biometric tasks such as liveness detection, bringing into question their use in security-critical settings in real life, such as phone banking.



## **11. PowerGAN: A Machine Learning Approach for Power Side-Channel Attack on Compute-in-Memory Accelerators**

cs.CR

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2304.11056v2) [paper-pdf](http://arxiv.org/pdf/2304.11056v2)

**Authors**: Ziyu Wang, Yuting Wu, Yongmo Park, Sangmin Yoo, Xinxin Wang, Jason K. Eshraghian, Wei D. Lu

**Abstract**: Analog compute-in-memory (CIM) systems are promising for deep neural network (DNN) inference acceleration due to their energy efficiency and high throughput. However, as the use of DNNs expands, protecting user input privacy has become increasingly important. In this paper, we identify a potential security vulnerability wherein an adversary can reconstruct the user's private input data from a power side-channel attack, under proper data acquisition and pre-processing, even without knowledge of the DNN model. We further demonstrate a machine learning-based attack approach using a generative adversarial network (GAN) to enhance the data reconstruction. Our results show that the attack methodology is effective in reconstructing user inputs from analog CIM accelerator power leakage, even at large noise levels and after countermeasures are applied. Specifically, we demonstrate the efficacy of our approach on an example of U-Net inference chip for brain tumor detection, and show the original magnetic resonance imaging (MRI) medical images can be successfully reconstructed even at a noise-level of 20% standard deviation of the maximum power signal value. Our study highlights a potential security vulnerability in analog CIM accelerators and raises awareness of using GAN to breach user privacy in such systems.



## **12. Two Heads are Better than One: Towards Better Adversarial Robustness by Combining Transduction and Rejection**

cs.LG

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17528v1) [paper-pdf](http://arxiv.org/pdf/2305.17528v1)

**Authors**: Nils Palumbo, Yang Guo, Xi Wu, Jiefeng Chen, Yingyu Liang, Somesh Jha

**Abstract**: Both transduction and rejection have emerged as important techniques for defending against adversarial perturbations. A recent work by Tram\`er showed that, in the rejection-only case (no transduction), a strong rejection-solution can be turned into a strong (but computationally inefficient) non-rejection solution. This detector-to-classifier reduction has been mostly applied to give evidence that certain claims of strong selective-model solutions are susceptible, leaving the benefits of rejection unclear. On the other hand, a recent work by Goldwasser et al. showed that rejection combined with transduction can give provable guarantees (for certain problems) that cannot be achieved otherwise. Nevertheless, under recent strong adversarial attacks (GMSA, which has been shown to be much more effective than AutoAttack against transduction), Goldwasser et al.'s work was shown to have low performance in a practical deep-learning setting. In this paper, we take a step towards realizing the promise of transduction+rejection in more realistic scenarios. Theoretically, we show that a novel application of Tram\`er's classifier-to-detector technique in the transductive setting can give significantly improved sample-complexity for robust generalization. While our theoretical construction is computationally inefficient, it guides us to identify an efficient transductive algorithm to learn a selective model. Extensive experiments using state of the art attacks (AutoAttack, GMSA) show that our solutions provide significantly better robust accuracy.



## **13. Backdooring Neural Code Search**

cs.SE

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17506v1) [paper-pdf](http://arxiv.org/pdf/2305.17506v1)

**Authors**: Weisong Sun, Yuchen Chen, Guanhong Tao, Chunrong Fang, Xiangyu Zhang, Quanjun Zhang, Bin Luo

**Abstract**: Reusing off-the-shelf code snippets from online repositories is a common practice, which significantly enhances the productivity of software developers. To find desired code snippets, developers resort to code search engines through natural language queries. Neural code search models are hence behind many such engines. These models are based on deep learning and gain substantial attention due to their impressive performance. However, the security aspect of these models is rarely studied. Particularly, an adversary can inject a backdoor in neural code search models, which return buggy or even vulnerable code with security/privacy issues. This may impact the downstream software (e.g., stock trading systems and autonomous driving) and cause financial loss and/or life-threatening incidents. In this paper, we demonstrate such attacks are feasible and can be quite stealthy. By simply modifying one variable/function name, the attacker can make buggy/vulnerable code rank in the top 11%. Our attack BADCODE features a special trigger generation and injection procedure, making the attack more effective and stealthy. The evaluation is conducted on two neural code search models and the results show our attack outperforms baselines by 60%. Our user study demonstrates that our attack is more stealthy than the baseline by two times based on the F1 score.



## **14. Modeling Adversarial Attack on Pre-trained Language Models as Sequential Decision Making**

cs.CL

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17440v1) [paper-pdf](http://arxiv.org/pdf/2305.17440v1)

**Authors**: Xuanjie Fang, Sijie Cheng, Yang Liu, Wei Wang

**Abstract**: Pre-trained language models (PLMs) have been widely used to underpin various downstream tasks. However, the adversarial attack task has found that PLMs are vulnerable to small perturbations. Mainstream methods adopt a detached two-stage framework to attack without considering the subsequent influence of substitution at each step. In this paper, we formally model the adversarial attack task on PLMs as a sequential decision-making problem, where the whole attack process is sequential with two decision-making problems, i.e., word finder and word substitution. Considering the attack process can only receive the final state without any direct intermediate signals, we propose to use reinforcement learning to find an appropriate sequential attack path to generate adversaries, named SDM-Attack. Extensive experimental results show that SDM-Attack achieves the highest attack success rate with a comparable modification rate and semantic similarity to attack fine-tuned BERT. Furthermore, our analyses demonstrate the generalization and transferability of SDM-Attack. The code is available at https://github.com/fduxuan/SDM-Attack.



## **15. On the Importance of Backbone to the Adversarial Robustness of Object Detectors**

cs.CV

12 pages

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17438v1) [paper-pdf](http://arxiv.org/pdf/2305.17438v1)

**Authors**: Xiao Li, Hang Chen, Xiaolin Hu

**Abstract**: Object detection is a critical component of various security-sensitive applications, such as autonomous driving and video surveillance. However, existing deep learning-based object detectors are vulnerable to adversarial attacks, which poses a significant challenge to their reliability and safety. Through experiments, we found that existing works on improving the adversarial robustness of object detectors have given a false sense of security. We argue that using adversarially pre-trained backbone networks is essential for enhancing the adversarial robustness of object detectors. We propose a simple yet effective recipe for fast adversarial fine-tuning on object detectors with adversarially pre-trained backbones. Without any modifications to the structure of object detectors, our recipe achieved significantly better adversarial robustness than previous works. Moreover, we explore the potential of different modern object detectors to improve adversarial robustness using our recipe and demonstrate several interesting findings. Our empirical results set a new milestone and deepen the understanding of adversarially robust object detection. Code and trained checkpoints will be publicly available.



## **16. Rethinking Adversarial Policies: A Generalized Attack Formulation and Provable Defense in Multi-Agent RL**

cs.LG

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17342v1) [paper-pdf](http://arxiv.org/pdf/2305.17342v1)

**Authors**: Xiangyu Liu, Souradip Chakraborty, Yanchao Sun, Furong Huang

**Abstract**: Most existing works consider direct perturbations of victim's state/action or the underlying transition dynamics to show vulnerability of reinforcement learning agents under adversarial attacks. However, such direct manipulation may not always be feasible in practice. In this paper, we consider another common and realistic attack setup: in a multi-agent RL setting with well-trained agents, during deployment time, the victim agent $\nu$ is exploited by an attacker who controls another agent $\alpha$ to act adversarially against the victim using an \textit{adversarial policy}. Prior attack models under such setup do not consider that the attacker can confront resistance and thus can only take partial control of the agent $\alpha$, as well as introducing perceivable ``abnormal'' behaviors that are easily detectable. A provable defense against these adversarial policies is also lacking. To resolve these issues, we introduce a more general attack formulation that models to what extent the adversary is able to control the agent to produce the adversarial policy. Based on such a generalized attack framework, the attacker can also regulate the state distribution shift caused by the attack through an attack budget, and thus produce stealthy adversarial policies that can exploit the victim agent. Furthermore, we provide the first provably robust defenses with convergence guarantee to the most robust victim policy via adversarial training with timescale separation, in sharp contrast to adversarial training in supervised learning which may only provide {\it empirical} defenses.



## **17. Certifiably Robust Reinforcement Learning through Model-Based Abstract Interpretation**

cs.LG

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2301.11374v2) [paper-pdf](http://arxiv.org/pdf/2301.11374v2)

**Authors**: Chenxi Yang, Greg Anderson, Swarat Chaudhuri

**Abstract**: We present a reinforcement learning (RL) framework in which the learned policy comes with a machine-checkable certificate of provable adversarial robustness. Our approach, called CAROL, learns a model of the environment. In each learning iteration, it uses the current version of this model and an external abstract interpreter to construct a differentiable signal for provable robustness. This signal is used to guide learning, and the abstract interpretation used to construct it directly leads to the robustness certificate returned at convergence. We give a theoretical analysis that bounds the worst-case accumulative reward of CAROL. We also experimentally evaluate CAROL on four MuJoCo environments with continuous state and action spaces. On these tasks, CAROL learns policies that, when contrasted with policies from the state-of-the-art robust RL algorithms, exhibit: (i) markedly enhanced certified performance lower bounds; and (ii) comparable performance under empirical adversarial attacks.



## **18. Adversarial Attacks on Online Learning to Rank with Click Feedback**

cs.LG

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2305.17071v1) [paper-pdf](http://arxiv.org/pdf/2305.17071v1)

**Authors**: Jinhang Zuo, Zhiyao Zhang, Zhiyong Wang, Shuai Li, Mohammad Hajiesmaili, Adam Wierman

**Abstract**: Online learning to rank (OLTR) is a sequential decision-making problem where a learning agent selects an ordered list of items and receives feedback through user clicks. Although potential attacks against OLTR algorithms may cause serious losses in real-world applications, little is known about adversarial attacks on OLTR. This paper studies attack strategies against multiple variants of OLTR. Our first result provides an attack strategy against the UCB algorithm on classical stochastic bandits with binary feedback, which solves the key issues caused by bounded and discrete feedback that previous works can not handle. Building on this result, we design attack algorithms against UCB-based OLTR algorithms in position-based and cascade models. Finally, we propose a general attack strategy against any algorithm under the general click model. Each attack algorithm manipulates the learning agent into choosing the target attack item $T-o(T)$ times, incurring a cumulative cost of $o(T)$. Experiments on synthetic and real data further validate the effectiveness of our proposed attack algorithms.



## **19. Leveraging characteristics of the output probability distribution for identifying adversarial audio examples**

cs.SD

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2305.17000v1) [paper-pdf](http://arxiv.org/pdf/2305.17000v1)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks represent a security threat to machine learning based automatic speech recognition (ASR) systems. To prevent such attacks we propose an adversarial example detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy, and the Jensen-Shannon divergence of the distributions of subsequent time steps. Then, we fit a Gaussian distribution to the characteristics observed for benign data. By computing the likelihood of incoming new audio we can distinguish malicious inputs from samples from clean data with an area under the receiving operator characteristic (AUROC) higher than 0.99, which drops to 0.98 for less-quality audio. To assess the robustness of our method we build adaptive attacks. This reduces the AUROC to 0.96 but results in more noisy adversarial clips.



## **20. Training Socially Aligned Language Models in Simulated Human Society**

cs.CL

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2305.16960v1) [paper-pdf](http://arxiv.org/pdf/2305.16960v1)

**Authors**: Ruibo Liu, Ruixin Yang, Chenyan Jia, Ge Zhang, Denny Zhou, Andrew M. Dai, Diyi Yang, Soroush Vosoughi

**Abstract**: Social alignment in AI systems aims to ensure that these models behave according to established societal values. However, unlike humans, who derive consensus on value judgments through social interaction, current language models (LMs) are trained to rigidly replicate their training corpus in isolation, leading to subpar generalization in unfamiliar scenarios and vulnerability to adversarial attacks. This work presents a novel training paradigm that permits LMs to learn from simulated social interactions. In comparison to existing methodologies, our approach is considerably more scalable and efficient, demonstrating superior performance in alignment benchmarks and human evaluations. This paradigm shift in the training of LMs brings us a step closer to developing AI systems that can robustly and accurately reflect societal norms and values.



## **21. Trust-Aware Resilient Control and Coordination of Connected and Automated Vehicles**

cs.MA

Keywords: Resilient control and coordination, Cybersecurity, Safety  guaranteed coordination, Connected And Autonomous Vehicles

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2305.16818v1) [paper-pdf](http://arxiv.org/pdf/2305.16818v1)

**Authors**: H M Sabbir Ahmad, Ehsan Sabouni, Wei Xiao, Christos G. Cassandras, Wenchao Li

**Abstract**: Security is crucial for cyber-physical systems, such as a network of Connected and Automated Vehicles (CAVs) cooperating to navigate through a road network safely. In this paper, we tackle the security of a cooperating network of CAVs in conflict areas by identifying the critical adversarial objectives from the point of view of uncooperative/malicious agents from our preliminary study, which are (i) safety violations resulting in collisions, and (ii) traffic jams. We utilize a trust framework (and our work doesn't depend on the specific choice of trust/reputation framework) to propose a resilient control and coordination framework that mitigates the effects of such agents and guarantees safe coordination. A class of attacks that can be used to achieve the adversarial objectives is Sybil attacks, which we use to validate our proposed framework through simulation studies. Besides that, we propose an attack detection and mitigation scheme using the trust framework. The simulation results demonstrate that our proposed scheme can detect fake CAVs during a Sybil attack, guarantee safe coordination, and mitigate their effects.



## **22. Robust Nonparametric Regression under Poisoning Attack**

math.ST

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2305.16771v1) [paper-pdf](http://arxiv.org/pdf/2305.16771v1)

**Authors**: Puning Zhao, Zhiguo Wan

**Abstract**: This paper studies robust nonparametric regression, in which an adversarial attacker can modify the values of up to $q$ samples from a training dataset of size $N$. Our initial solution is an M-estimator based on Huber loss minimization. Compared with simple kernel regression, i.e. the Nadaraya-Watson estimator, this method can significantly weaken the impact of malicious samples on the regression performance. We provide the convergence rate as well as the corresponding minimax lower bound. The result shows that, with proper bandwidth selection, $\ell_\infty$ error is minimax optimal. The $\ell_2$ error is optimal if $q\lesssim \sqrt{N/\ln^2 N}$, but is suboptimal with larger $q$. The reason is that this estimator is vulnerable if there are many attacked samples concentrating in a small region. To address this issue, we propose a correction method by projecting the initial estimate to the space of Lipschitz functions. The final estimate is nearly minimax optimal for arbitrary $q$, up to a $\ln N$ factor.



## **23. Mitigating Adversarial Attacks by Distributing Different Copies to Different Users**

cs.CR

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2111.15160v3) [paper-pdf](http://arxiv.org/pdf/2111.15160v3)

**Authors**: Jiyi Zhang, Han Fang, Wesley Joon-Wie Tann, Ke Xu, Chengfang Fang, Ee-Chien Chang

**Abstract**: Machine learning models are vulnerable to adversarial attacks. In this paper, we consider the scenario where a model is distributed to multiple buyers, among which a malicious buyer attempts to attack another buyer. The malicious buyer probes its copy of the model to search for adversarial samples and then presents the found samples to the victim's copy of the model in order to replicate the attack. We point out that by distributing different copies of the model to different buyers, we can mitigate the attack such that adversarial samples found on one copy would not work on another copy. We observed that training a model with different randomness indeed mitigates such replication to a certain degree. However, there is no guarantee and retraining is computationally expensive. A number of works extended the retraining method to enhance the differences among models. However, a very limited number of models can be produced using such methods and the computational cost becomes even higher. Therefore, we propose a flexible parameter rewriting method that directly modifies the model's parameters. This method does not require additional training and is able to generate a large number of copies in a more controllable manner, where each copy induces different adversarial regions. Experimentation studies show that rewriting can significantly mitigate the attacks while retaining high classification accuracy. For instance, on GTSRB dataset with respect to Hop Skip Jump attack, using attractor-based rewriter can reduce the success rate of replicating the attack to 0.5% while independently training copies with different randomness can reduce the success rate to 6.5%. From this study, we believe that there are many further directions worth exploring.



## **24. IMBERT: Making BERT Immune to Insertion-based Backdoor Attacks**

cs.CL

accepted to Third Workshop on Trustworthy Natural Language Processing

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.16503v1) [paper-pdf](http://arxiv.org/pdf/2305.16503v1)

**Authors**: Xuanli He, Jun Wang, Benjamin Rubinstein, Trevor Cohn

**Abstract**: Backdoor attacks are an insidious security threat against machine learning models. Adversaries can manipulate the predictions of compromised models by inserting triggers into the training phase. Various backdoor attacks have been devised which can achieve nearly perfect attack success without affecting model predictions for clean inputs. Means of mitigating such vulnerabilities are underdeveloped, especially in natural language processing. To fill this gap, we introduce IMBERT, which uses either gradients or self-attention scores derived from victim models to self-defend against backdoor attacks at inference time. Our empirical studies demonstrate that IMBERT can effectively identify up to 98.5% of inserted triggers. Thus, it significantly reduces the attack success rate while attaining competitive accuracy on the clean dataset across widespread insertion-based attacks compared to two baselines. Finally, we show that our approach is model-agnostic, and can be easily ported to several pre-trained transformer models.



## **25. Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability**

cs.CV

code repo: https://github.com/xavihart/Diff-PGD

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.16494v1) [paper-pdf](http://arxiv.org/pdf/2305.16494v1)

**Authors**: Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen

**Abstract**: Neural networks are known to be susceptible to adversarial samples: small variations of natural examples crafted to deliberately mislead the models. While they can be easily generated using gradient-based techniques in digital and physical scenarios, they often differ greatly from the actual data distribution of natural images, resulting in a trade-off between strength and stealthiness. In this paper, we propose a novel framework dubbed Diffusion-Based Projected Gradient Descent (Diff-PGD) for generating realistic adversarial samples. By exploiting a gradient guided by a diffusion model, Diff-PGD ensures that adversarial samples remain close to the original data distribution while maintaining their effectiveness. Moreover, our framework can be easily customized for specific tasks such as digital attacks, physical-world attacks, and style-based attacks. Compared with existing methods for generating natural-style adversarial samples, our framework enables the separation of optimizing adversarial loss from other surrogate losses (e.g., content/smoothness/style loss), making it more stable and controllable. Finally, we demonstrate that the samples generated using Diff-PGD have better transferability and anti-purification power than traditional gradient-based methods. Code will be released in https://github.com/xavihart/Diff-PGD



## **26. Path Defense in Dynamic Defender-Attacker Blotto Games (dDAB) with Limited Information**

cs.GT

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2204.04176v2) [paper-pdf](http://arxiv.org/pdf/2204.04176v2)

**Authors**: Austin K. Chen, Bryce L. Ferguson, Daigo Shishika, Michael Dorothy, Jason R. Marden, George J. Pappas, Vijay Kumar

**Abstract**: We consider a path guarding problem in dynamic Defender-Attacker Blotto games (dDAB), where a team of robots must defend a path in a graph against adversarial agents. Multi-robot systems are particularly well suited to this application, as recent work has shown the effectiveness of these systems in related areas such as perimeter defense and surveillance. When designing a defender policy that guarantees the defense of a path, information about the adversary and the environment can be helpful and may reduce the number of resources required by the defender to achieve a sufficient level of security. In this work, we characterize the necessary and sufficient number of assets needed to guarantee the defense of a shortest path between two nodes in dDAB games when the defender can only detect assets within $k$-hops of a shortest path. By characterizing the relationship between sensing horizon and required resources, we show that increasing the sensing capability of the defender greatly reduces the number of defender assets needed to defend the path.



## **27. Don't Retrain, Just Rewrite: Countering Adversarial Perturbations by Rewriting Text**

cs.CL

Accepted to ACL 2023

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.16444v1) [paper-pdf](http://arxiv.org/pdf/2305.16444v1)

**Authors**: Ashim Gupta, Carter Wood Blum, Temma Choji, Yingjie Fei, Shalin Shah, Alakananda Vempala, Vivek Srikumar

**Abstract**: Can language models transform inputs to protect text classifiers against adversarial attacks? In this work, we present ATINTER, a model that intercepts and learns to rewrite adversarial inputs to make them non-adversarial for a downstream text classifier. Our experiments on four datasets and five attack mechanisms reveal that ATINTER is effective at providing better adversarial robustness than existing defense approaches, without compromising task accuracy. For example, on sentiment classification using the SST-2 dataset, our method improves the adversarial accuracy over the best existing defense approach by more than 4% with a smaller decrease in task accuracy (0.5% vs 2.5%). Moreover, we show that ATINTER generalizes across multiple downstream tasks and classifiers without having to explicitly retrain it for those settings. Specifically, we find that when ATINTER is trained to remove adversarial perturbations for the sentiment classification task on the SST-2 dataset, it even transfers to a semantically different task of news classification (on AGNews) and improves the adversarial robustness by more than 10%.



## **28. On the Robustness of Segment Anything**

cs.CV

22 pages

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.16220v1) [paper-pdf](http://arxiv.org/pdf/2305.16220v1)

**Authors**: Yihao Huang, Yue Cao, Tianlin Li, Felix Juefei-Xu, Di Lin, Ivor W. Tsang, Yang Liu, Qing Guo

**Abstract**: Segment anything model (SAM) has presented impressive objectness identification capability with the idea of prompt learning and a new collected large-scale dataset. Given a prompt (e.g., points, bounding boxes, or masks) and an input image, SAM is able to generate valid segment masks for all objects indicated by the prompts, presenting high generalization across diverse scenarios and being a general method for zero-shot transfer to downstream vision tasks. Nevertheless, it remains unclear whether SAM may introduce errors in certain threatening scenarios. Clarifying this is of significant importance for applications that require robustness, such as autonomous vehicles. In this paper, we aim to study the testing-time robustness of SAM under adversarial scenarios and common corruptions. To this end, we first build a testing-time robustness evaluation benchmark for SAM by integrating existing public datasets. Second, we extend representative adversarial attacks against SAM and study the influence of different prompts on robustness. Third, we study the robustness of SAM under diverse corruption types by evaluating SAM on corrupted datasets with different prompts. With experiments conducted on SA-1B and KITTI datasets, we find that SAM exhibits remarkable robustness against various corruptions, except for blur-related corruption. Furthermore, SAM remains susceptible to adversarial attacks, particularly when subjected to PGD and BIM attacks. We think such a comprehensive study could highlight the importance of the robustness issues of SAM and trigger a series of new tasks for SAM as well as downstream vision tasks.



## **29. ByzSecAgg: A Byzantine-Resistant Secure Aggregation Scheme for Federated Learning Based on Coded Computing and Vector Commitment**

cs.CR

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2302.09913v2) [paper-pdf](http://arxiv.org/pdf/2302.09913v2)

**Authors**: Tayyebeh Jahani-Nezhad, Mohammad Ali Maddah-Ali, Giuseppe Caire

**Abstract**: In this paper, we propose an efficient secure aggregation scheme for federated learning that is protected against Byzantine attacks and privacy leakages. Processing individual updates to manage adversarial behavior, while preserving privacy of data against colluding nodes, requires some sort of secure secret sharing. However, communication load for secret sharing of long vectors of updates can be very high. To resolve this issue, in the proposed scheme, local updates are partitioned into smaller sub-vectors and shared using ramp secret sharing. However, this sharing method does not admit bi-linear computations, such as pairwise distance calculations, needed by outlier-detection algorithms. To overcome this issue, each user runs another round of ramp sharing, with different embedding of data in the sharing polynomial. This technique, motivated by ideas from coded computing, enables secure computation of pairwise distance. In addition, to maintain the integrity and privacy of the local update, the proposed scheme also uses a vector commitment method, in which the commitment size remains constant (i.e. does not increase with the length of the local update), while simultaneously allowing verification of the secret sharing process.



## **30. Impact of Adversarial Training on Robustness and Generalizability of Language Models**

cs.CL

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2211.05523v2) [paper-pdf](http://arxiv.org/pdf/2211.05523v2)

**Authors**: Enes Altinisik, Hassan Sajjad, Husrev Taha Sencar, Safa Messaoud, Sanjay Chawla

**Abstract**: Adversarial training is widely acknowledged as the most effective defense against adversarial attacks. However, it is also well established that achieving both robustness and generalization in adversarially trained models involves a trade-off. The goal of this work is to provide an in depth comparison of different approaches for adversarial training in language models. Specifically, we study the effect of pre-training data augmentation as well as training time input perturbations vs. embedding space perturbations on the robustness and generalization of transformer-based language models. Our findings suggest that better robustness can be achieved by pre-training data augmentation or by training with input space perturbation. However, training with embedding space perturbation significantly improves generalization. A linguistic correlation analysis of neurons of the learned models reveals that the improved generalization is due to 'more specialized' neurons. To the best of our knowledge, this is the first work to carry out a deep qualitative analysis of different methods of generating adversarial examples in adversarial training of language models.



## **31. IDEA: Invariant Causal Defense for Graph Adversarial Robustness**

cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15792v1) [paper-pdf](http://arxiv.org/pdf/2305.15792v1)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Bingbing Xu, Xueqi Cheng

**Abstract**: Graph neural networks (GNNs) have achieved remarkable success in various tasks, however, their vulnerability to adversarial attacks raises concerns for the real-world applications. Existing defense methods can resist some attacks, but suffer unbearable performance degradation under other unknown attacks. This is due to their reliance on either limited observed adversarial examples to optimize (adversarial training) or specific heuristics to alter graph or model structures (graph purification or robust aggregation). In this paper, we propose an Invariant causal DEfense method against adversarial Attacks (IDEA), providing a new perspective to address this issue. The method aims to learn causal features that possess strong predictability for labels and invariant predictability across attacks, to achieve graph adversarial robustness. Through modeling and analyzing the causal relationships in graph adversarial attacks, we design two invariance objectives to learn the causal features. Extensive experiments demonstrate that our IDEA significantly outperforms all the baselines under both poisoning and evasion attacks on five benchmark datasets, highlighting the strong and invariant predictability of IDEA. The implementation of IDEA is available at https://anonymous.4open.science/r/IDEA_repo-666B.



## **32. Healing Unsafe Dialogue Responses with Weak Supervision Signals**

cs.CL

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15757v1) [paper-pdf](http://arxiv.org/pdf/2305.15757v1)

**Authors**: Zi Liang, Pinghui Wang, Ruofei Zhang, Shuo Zhang, Xiaofan Ye Yi Huang, Junlan Feng

**Abstract**: Recent years have seen increasing concerns about the unsafe response generation of large-scale dialogue systems, where agents will learn offensive or biased behaviors from the real-world corpus. Some methods are proposed to address the above issue by detecting and replacing unsafe training examples in a pipeline style. Though effective, they suffer from a high annotation cost and adapt poorly to unseen scenarios as well as adversarial attacks. Besides, the neglect of providing safe responses (e.g. simply replacing with templates) will cause the information-missing problem of dialogues. To address these issues, we propose an unsupervised pseudo-label sampling method, TEMP, that can automatically assign potential safe responses. Specifically, our TEMP method groups responses into several clusters and samples multiple labels with an adaptively sharpened sampling strategy, inspired by the observation that unsafe samples in the clusters are usually few and distribute in the tail. Extensive experiments in chitchat and task-oriented dialogues show that our TEMP outperforms state-of-the-art models with weak supervision signals and obtains comparable results under unsupervised learning settings.



## **33. PEARL: Preprocessing Enhanced Adversarial Robust Learning of Image Deraining for Semantic Segmentation**

cs.CV

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15709v1) [paper-pdf](http://arxiv.org/pdf/2305.15709v1)

**Authors**: Xianghao Jiao, Yaohua Liu, Jiaxin Gao, Xinyuan Chu, Risheng Liu, Xin Fan

**Abstract**: In light of the significant progress made in the development and application of semantic segmentation tasks, there has been increasing attention towards improving the robustness of segmentation models against natural degradation factors (e.g., rain streaks) or artificially attack factors (e.g., adversarial attack). Whereas, most existing methods are designed to address a single degradation factor and are tailored to specific application scenarios. In this work, we present the first attempt to improve the robustness of semantic segmentation tasks by simultaneously handling different types of degradation factors. Specifically, we introduce the Preprocessing Enhanced Adversarial Robust Learning (PEARL) framework based on the analysis of our proposed Naive Adversarial Training (NAT) framework. Our approach effectively handles both rain streaks and adversarial perturbation by transferring the robustness of the segmentation model to the image derain model. Furthermore, as opposed to the commonly used Negative Adversarial Attack (NAA), we design the Auxiliary Mirror Attack (AMA) to introduce positive information prior to the training of the PEARL framework, which improves defense capability and segmentation performance. Our extensive experiments and ablation studies based on different derain methods and segmentation models have demonstrated the significant performance improvement of PEARL with AMA in defense against various adversarial attacks and rain streaks while maintaining high generalization performance across different datasets.



## **34. Rethink Diversity in Deep Learning Testing**

cs.SE

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15698v1) [paper-pdf](http://arxiv.org/pdf/2305.15698v1)

**Authors**: Zi Wang, Jihye Choi, Somesh Jha

**Abstract**: Deep neural networks (DNNs) have demonstrated extraordinary capabilities and are an integral part of modern software systems. However, they also suffer from various vulnerabilities such as adversarial attacks and unfairness. Testing deep learning (DL) systems is therefore an important task, to detect and mitigate those vulnerabilities. Motivated by the success of traditional software testing, which often employs diversity heuristics, various diversity measures on DNNs have been proposed to help efficiently expose the buggy behavior of DNNs. In this work, we argue that many DNN testing tasks should be treated as directed testing problems rather than general-purpose testing tasks, because these tasks are specific and well-defined. Hence, the diversity-based approach is less effective.   Following our argument based on the semantics of DNNs and the testing goal, we derive $6$ metrics that can be used for DNN testing and carefully analyze their application scopes. We empirically show their efficacy in exposing bugs in DNNs compared to recent diversity-based metrics. Moreover, we also notice discrepancies between the practices of the software engineering (SE) community and the DL community. We point out some of these gaps, and hopefully, this can lead to bridging the SE practice and DL findings.



## **35. AdvFunMatch: When Consistent Teaching Meets Adversarial Robustness**

cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.14700v2) [paper-pdf](http://arxiv.org/pdf/2305.14700v2)

**Authors**: Zihui Wu, Haichang Gao, Bingqian Zhou, Ping Wang

**Abstract**: \emph{Consistent teaching} is an effective paradigm for implementing knowledge distillation (KD), where both student and teacher models receive identical inputs, and KD is treated as a function matching task (FunMatch). However, one limitation of FunMatch is that it does not account for the transfer of adversarial robustness, a model's resistance to adversarial attacks. To tackle this problem, we propose a simple but effective strategy called Adversarial Function Matching (AdvFunMatch), which aims to match distributions for all data points within the $\ell_p$-norm ball of the training data, in accordance with consistent teaching. Formulated as a min-max optimization problem, AdvFunMatch identifies the worst-case instances that maximizes the KL-divergence between teacher and student model outputs, which we refer to as "mismatched examples," and then matches the outputs on these mismatched examples. Our experimental results show that AdvFunMatch effectively produces student models with both high clean accuracy and robustness. Furthermore, we reveal that strong data augmentations (\emph{e.g.}, AutoAugment) are beneficial in AdvFunMatch, whereas prior works have found them less effective in adversarial training. Code is available at \url{https://gitee.com/zihui998/adv-fun-match}.



## **36. Near Optimal Adversarial Attack on UCB Bandits**

cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2008.09312v4) [paper-pdf](http://arxiv.org/pdf/2008.09312v4)

**Authors**: Shiliang Zuo

**Abstract**: I study a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. I propose a novel attack strategy that manipulates a learner employing the UCB algorithm into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\widehat{O}(\sqrt{\log T})$, where $T$ is the number of rounds. I also prove the first lower bound on the cumulative attack cost. The lower bound matches the upper bound up to $O(\log \log T)$ factors, showing the proposed attack strategy to be near optimal.



## **37. How do humans perceive adversarial text? A reality check on the validity and naturalness of word-based adversarial attacks**

cs.CL

ACL 2023

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15587v1) [paper-pdf](http://arxiv.org/pdf/2305.15587v1)

**Authors**: Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy

**Abstract**: Natural Language Processing (NLP) models based on Machine Learning (ML) are susceptible to adversarial attacks -- malicious algorithms that imperceptibly modify input text to force models into making incorrect predictions. However, evaluations of these attacks ignore the property of imperceptibility or study it under limited settings. This entails that adversarial perturbations would not pass any human quality gate and do not represent real threats to human-checked NLP systems. To bypass this limitation and enable proper assessment (and later, improvement) of NLP model robustness, we have surveyed 378 human participants about the perceptibility of text adversarial examples produced by state-of-the-art methods. Our results underline that existing text attacks are impractical in real-world scenarios where humans are involved. This contrasts with previous smaller-scale human studies, which reported overly optimistic conclusions regarding attack success. Through our work, we hope to position human perceptibility as a first-class success criterion for text attacks, and provide guidance for research to build effective attack algorithms and, in turn, design appropriate defence mechanisms.



## **38. Non-Asymptotic Lower Bounds For Training Data Reconstruction**

cs.LG

Additional experiments and minor bug fixes

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2303.16372v4) [paper-pdf](http://arxiv.org/pdf/2303.16372v4)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: Mathematical notions of privacy, such as differential privacy, are often stated as probabilistic guarantees that are difficult to interpret. It is imperative, however, that the implications of data sharing be effectively communicated to the data principal to ensure informed decision-making and offer full transparency with regards to the associated privacy risks. To this end, our work presents a rigorous quantitative evaluation of the protection conferred by private learners by investigating their resilience to training data reconstruction attacks. We accomplish this by deriving non-asymptotic lower bounds on the reconstruction error incurred by any adversary against $(\epsilon, \delta)$ differentially private learners for target samples that belong to any compact metric space. Working with a generalization of differential privacy, termed metric privacy, we remove boundedness assumptions on the input space prevalent in prior work, and prove that our results hold for general locally compact metric spaces. We extend the analysis to cover the high dimensional regime, wherein, the input data dimensionality may be larger than the adversary's query budget, and demonstrate that our bounds are minimax optimal under certain regimes.



## **39. Fast Adversarial CNN-based Perturbation Attack on No-Reference Image- and Video-Quality Metrics**

cs.CV

ICLR 2023 TinyPapers

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15544v1) [paper-pdf](http://arxiv.org/pdf/2305.15544v1)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Modern neural-network-based no-reference image- and video-quality metrics exhibit performance as high as full-reference metrics. These metrics are widely used to improve visual quality in computer vision methods and compare video processing methods. However, these metrics are not stable to traditional adversarial attacks, which can cause incorrect results. Our goal is to investigate the boundaries of no-reference metrics applicability, and in this paper, we propose a fast adversarial perturbation attack on no-reference quality metrics. The proposed attack (FACPA) can be exploited as a preprocessing step in real-time video processing and compression algorithms. This research can yield insights to further aid in designing of stable neural-network-based no-reference quality metrics.



## **40. Robust Classification via a Single Diffusion Model**

cs.CV

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15241v1) [paper-pdf](http://arxiv.org/pdf/2305.15241v1)

**Authors**: Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, Jun Zhu

**Abstract**: Recently, diffusion models have been successfully applied to improving adversarial robustness of image classifiers by purifying the adversarial noises or generating realistic data for adversarial training. However, the diffusion-based purification can be evaded by stronger adaptive attacks while adversarial training does not perform well under unseen threats, exhibiting inevitable limitations of these methods. To better harness the expressive power of diffusion models, in this paper we propose Robust Diffusion Classifier (RDC), a generative classifier that is constructed from a pre-trained diffusion model to be adversarially robust. Our method first maximizes the data likelihood of a given input and then predicts the class probabilities of the optimized input using the conditional likelihood of the diffusion model through Bayes' theorem. Since our method does not require training on particular adversarial attacks, we demonstrate that it is more generalizable to defend against multiple unseen threats. In particular, RDC achieves $73.24\%$ robust accuracy against $\ell_\infty$ norm-bounded perturbations with $\epsilon_\infty=8/255$ on CIFAR-10, surpassing the previous state-of-the-art adversarial training models by $+2.34\%$. The findings highlight the potential of generative classifiers by employing diffusion models for adversarial robustness compared with the commonly studied discriminative classifiers.



## **41. Adaptive Data Analysis in a Balanced Adversarial Model**

cs.LG

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15452v1) [paper-pdf](http://arxiv.org/pdf/2305.15452v1)

**Authors**: Kobbi Nissim, Uri Stemmer, Eliad Tsfadia

**Abstract**: In adaptive data analysis, a mechanism gets $n$ i.i.d. samples from an unknown distribution $D$, and is required to provide accurate estimations to a sequence of adaptively chosen statistical queries with respect to $D$. Hardt and Ullman (FOCS 2014) and Steinke and Ullman (COLT 2015) showed that in general, it is computationally hard to answer more than $\Theta(n^2)$ adaptive queries, assuming the existence of one-way functions.   However, these negative results strongly rely on an adversarial model that significantly advantages the adversarial analyst over the mechanism, as the analyst, who chooses the adaptive queries, also chooses the underlying distribution $D$. This imbalance raises questions with respect to the applicability of the obtained hardness results -- an analyst who has complete knowledge of the underlying distribution $D$ would have little need, if at all, to issue statistical queries to a mechanism which only holds a finite number of samples from $D$.   We consider more restricted adversaries, called \emph{balanced}, where each such adversary consists of two separated algorithms: The \emph{sampler} who is the entity that chooses the distribution and provides the samples to the mechanism, and the \emph{analyst} who chooses the adaptive queries, but does not have a prior knowledge of the underlying distribution. We improve the quality of previous lower bounds by revisiting them using an efficient \emph{balanced} adversary, under standard public-key cryptography assumptions. We show that these stronger hardness assumptions are unavoidable in the sense that any computationally bounded \emph{balanced} adversary that has the structure of all known attacks, implies the existence of public-key cryptography.



## **42. Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension**

cs.LG

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15203v1) [paper-pdf](http://arxiv.org/pdf/2305.15203v1)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto D'Onofrio, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification, neural networks are known to be vulnerable to adversarial attacks. These attacks are small perturbations of the input data designed to fool the model. Naturally, a question arises regarding the potential connection between the architecture, settings, or properties of the model and the nature of the attack. In this work, we aim to shed light on this problem by focusing on the implicit bias of the neural network, which refers to its inherent inclination to favor specific patterns or outcomes. Specifically, we investigate one aspect of the implicit bias, which involves the essential Fourier frequencies required for accurate image classification. We conduct tests to assess the statistical relationship between these frequencies and those necessary for a successful attack. To delve into this relationship, we propose a new method that can uncover non-linear correlations between sets of coordinates, which, in our case, are the aforementioned frequencies. By exploiting the entanglement between intrinsic dimension and correlation, we provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are closely tied.



## **43. IoT Threat Detection Testbed Using Generative Adversarial Networks**

cs.CR

8 pages, 5 figures

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15191v1) [paper-pdf](http://arxiv.org/pdf/2305.15191v1)

**Authors**: Farooq Shaikh, Elias Bou-Harb, Aldin Vehabovic, Jorge Crichigno, Aysegul Yayimli, Nasir Ghani

**Abstract**: The Internet of Things(IoT) paradigm provides persistent sensing and data collection capabilities and is becoming increasingly prevalent across many market sectors. However, most IoT devices emphasize usability and function over security, making them very vulnerable to malicious exploits. This concern is evidenced by the increased use of compromised IoT devices in large scale bot networks (botnets) to launch distributed denial of service(DDoS) attacks against high value targets. Unsecured IoT systems can also provide entry points to private networks, allowing adversaries relatively easy access to valuable resources and services. Indeed, these evolving IoT threat vectors (ranging from brute force attacks to remote code execution exploits) are posing key challenges. Moreover, many traditional security mechanisms are not amenable for deployment on smaller resource-constrained IoT platforms. As a result, researchers have been developing a range of methods for IoT security, with many strategies using advanced machine learning(ML) techniques. Along these lines, this paper presents a novel generative adversarial network(GAN) solution to detect threats from malicious IoT devices both inside and outside a network. This model is trained using both benign IoT traffic and global darknet data and further evaluated in a testbed with real IoT devices and malware threats.



## **44. Another Dead End for Morphological Tags? Perturbed Inputs and Parsing**

cs.CL

Accepted at Findings of ACL 2023

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15119v1) [paper-pdf](http://arxiv.org/pdf/2305.15119v1)

**Authors**: Alberto Muñoz-Ortiz, David Vilares

**Abstract**: The usefulness of part-of-speech tags for parsing has been heavily questioned due to the success of word-contextualized parsers. Yet, most studies are limited to coarse-grained tags and high quality written content; while we know little about their influence when it comes to models in production that face lexical errors. We expand these setups and design an adversarial attack to verify if the use of morphological information by parsers: (i) contributes to error propagation or (ii) if on the other hand it can play a role to correct mistakes that word-only neural parsers make. The results on 14 diverse UD treebanks show that under such attacks, for transition- and graph-based models their use contributes to degrade the performance even faster, while for the (lower-performing) sequence labeling parsers they are helpful. We also show that if morphological tags were utopically robust against lexical perturbations, they would be able to correct parsing mistakes.



## **45. Adversarial Demonstration Attacks on Large Language Models**

cs.CL

Work in Progress

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14950v1) [paper-pdf](http://arxiv.org/pdf/2305.14950v1)

**Authors**: Jiongxiao Wang, Zichen Liu, Keun Hee Park, Muhao Chen, Chaowei Xiao

**Abstract**: With the emergence of more powerful large language models (LLMs), such as ChatGPT and GPT-4, in-context learning (ICL) has gained significant prominence in leveraging these models for specific tasks by utilizing data-label pairs as precondition prompts. While incorporating demonstrations can greatly enhance the performance of LLMs across various tasks, it may introduce a new security concern: attackers can manipulate only the demonstrations without changing the input to perform an attack. In this paper, we investigate the security concern of ICL from an adversarial perspective, focusing on the impact of demonstrations. We propose an ICL attack based on TextAttack, which aims to only manipulate the demonstration without changing the input to mislead the models. Our results demonstrate that as the number of demonstrations increases, the robustness of in-context learning would decreases. Furthermore, we also observe that adversarially attacked demonstrations exhibit transferability to diverse input examples. These findings emphasize the critical security risks associated with ICL and underscore the necessity for extensive research on the robustness of ICL, particularly given its increasing significance in the advancement of LLMs.



## **46. Madvex: Instrumentation-based Adversarial Attacks on Machine Learning Malware Detection**

cs.CR

20 pages. To be published in The 20th Conference on Detection of  Intrusions and Malware & Vulnerability Assessment (DIMVA 2023)

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.02559v2) [paper-pdf](http://arxiv.org/pdf/2305.02559v2)

**Authors**: Nils Loose, Felix Mächtle, Claudius Pott, Volodymyr Bezsmertnyi, Thomas Eisenbarth

**Abstract**: WebAssembly (Wasm) is a low-level binary format for web applications, which has found widespread adoption due to its improved performance and compatibility with existing software. However, the popularity of Wasm has also led to its exploitation for malicious purposes, such as cryptojacking, where malicious actors use a victim's computing resources to mine cryptocurrencies without their consent. To counteract this threat, machine learning-based detection methods aiming to identify cryptojacking activities within Wasm code have emerged. It is well-known that neural networks are susceptible to adversarial attacks, where inputs to a classifier are perturbed with minimal changes that result in a crass misclassification. While applying changes in image classification is easy, manipulating binaries in an automated fashion to evade malware classification without changing functionality is non-trivial. In this work, we propose a new approach to include adversarial examples in the code section of binaries via instrumentation. The introduced gadgets allow for the inclusion of arbitrary bytes, enabling efficient adversarial attacks that reliably bypass state-of-the-art machine learning classifiers such as the CNN-based Minos recently proposed at NDSS 2021. We analyze the cost and reliability of instrumentation-based adversarial example generation and show that the approach works reliably at minimal size and performance overheads.



## **47. Introducing Competition to Boost the Transferability of Targeted Adversarial Examples through Clean Feature Mixup**

cs.CV

CVPR 2023 camera-ready

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14846v1) [paper-pdf](http://arxiv.org/pdf/2305.14846v1)

**Authors**: Junyoung Byun, Myung-Joon Kwon, Seungju Cho, Yoonji Kim, Changick Kim

**Abstract**: Deep neural networks are widely known to be susceptible to adversarial examples, which can cause incorrect predictions through subtle input modifications. These adversarial examples tend to be transferable between models, but targeted attacks still have lower attack success rates due to significant variations in decision boundaries. To enhance the transferability of targeted adversarial examples, we propose introducing competition into the optimization process. Our idea is to craft adversarial perturbations in the presence of two new types of competitor noises: adversarial perturbations towards different target classes and friendly perturbations towards the correct class. With these competitors, even if an adversarial example deceives a network to extract specific features leading to the target class, this disturbance can be suppressed by other competitors. Therefore, within this competition, adversarial examples should take different attack strategies by leveraging more diverse features to overwhelm their interference, leading to improving their transferability to different models. Considering the computational complexity, we efficiently simulate various interference from these two types of competitors in feature space by randomly mixing up stored clean features in the model inference and named this method Clean Feature Mixup (CFM). Our extensive experimental results on the ImageNet-Compatible and CIFAR-10 datasets show that the proposed method outperforms the existing baselines with a clear margin. Our code is available at https://github.com/dreamflake/CFM.



## **48. Block Coordinate Descent on Smooth Manifolds**

math.OC

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14744v1) [paper-pdf](http://arxiv.org/pdf/2305.14744v1)

**Authors**: Liangzu Peng, René Vidal

**Abstract**: Block coordinate descent is an optimization paradigm that iteratively updates one block of variables at a time, making it quite amenable to big data applications due to its scalability and performance. Its convergence behavior has been extensively studied in the (block-wise) convex case, but it is much less explored in the non-convex case. In this paper we analyze the convergence of block coordinate methods on non-convex sets and derive convergence rates on smooth manifolds under natural or weaker assumptions than prior work. Our analysis applies to many non-convex problems (e.g., generalized PCA, optimal transport, matrix factorization, Burer-Monteiro factorization, outlier-robust estimation, alternating projection, maximal coding rate reduction, neural collapse, adversarial attacks, homomorphic sensing), either yielding novel corollaries or recovering previously known results.



## **49. Adversarial Machine Learning and Cybersecurity: Risks, Challenges, and Legal Implications**

cs.CR

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.14553v1) [paper-pdf](http://arxiv.org/pdf/2305.14553v1)

**Authors**: Micah Musser, Andrew Lohn, James X. Dempsey, Jonathan Spring, Ram Shankar Siva Kumar, Brenda Leong, Christina Liaghati, Cindy Martinez, Crystal D. Grant, Daniel Rohrer, Heather Frase, Jonathan Elliott, John Bansemer, Mikel Rodriguez, Mitt Regan, Rumman Chowdhury, Stefan Hermanek

**Abstract**: In July 2022, the Center for Security and Emerging Technology (CSET) at Georgetown University and the Program on Geopolitics, Technology, and Governance at the Stanford Cyber Policy Center convened a workshop of experts to examine the relationship between vulnerabilities in artificial intelligence systems and more traditional types of software vulnerabilities. Topics discussed included the extent to which AI vulnerabilities can be handled under standard cybersecurity processes, the barriers currently preventing the accurate sharing of information about AI vulnerabilities, legal issues associated with adversarial attacks on AI systems, and potential areas where government support could improve AI vulnerability management and mitigation.   This report is meant to accomplish two things. First, it provides a high-level discussion of AI vulnerabilities, including the ways in which they are disanalogous to other types of vulnerabilities, and the current state of affairs regarding information sharing and legal oversight of AI vulnerabilities. Second, it attempts to articulate broad recommendations as endorsed by the majority of participants at the workshop.



## **50. Translate your gibberish: black-box adversarial attack on machine translation systems**

cs.CL

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2303.10974v2) [paper-pdf](http://arxiv.org/pdf/2303.10974v2)

**Authors**: Andrei Chertkov, Olga Tsymboi, Mikhail Pautov, Ivan Oseledets

**Abstract**: Neural networks are deployed widely in natural language processing tasks on the industrial scale, and perhaps the most often they are used as compounds of automatic machine translation systems. In this work, we present a simple approach to fool state-of-the-art machine translation tools in the task of translation from Russian to English and vice versa. Using a novel black-box gradient-free tensor-based optimizer, we show that many online translation tools, such as Google, DeepL, and Yandex, may both produce wrong or offensive translations for nonsensical adversarial input queries and refuse to translate seemingly benign input phrases. This vulnerability may interfere with understanding a new language and simply worsen the user's experience while using machine translation systems, and, hence, additional improvements of these tools are required to establish better translation.



