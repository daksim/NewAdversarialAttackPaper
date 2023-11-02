# Latest Adversarial Attack Papers
**update at 2023-11-02 10:26:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Robustness Tests for Automatic Machine Translation Metrics with Adversarial Attacks**

cs.CL

Accepted in Findings of EMNLP 2023

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00508v1) [paper-pdf](http://arxiv.org/pdf/2311.00508v1)

**Authors**: Yichen Huang, Timothy Baldwin

**Abstract**: We investigate MT evaluation metric performance on adversarially-synthesized texts, to shed light on metric robustness. We experiment with word- and character-level attacks on three popular machine translation metrics: BERTScore, BLEURT, and COMET. Our human experiments validate that automatic metrics tend to overpenalize adversarially-degraded translations. We also identify inconsistencies in BERTScore ratings, where it judges the original sentence and the adversarially-degraded one as similar, while judging the degraded translation as notably worse than the original with respect to the reference. We identify patterns of brittleness that motivate more robust metric development.



## **2. Improving Robustness for Vision Transformer with a Simple Dynamic Scanning Augmentation**

cs.CV

Accepted in Neurocomputing

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00441v1) [paper-pdf](http://arxiv.org/pdf/2311.00441v1)

**Authors**: Shashank Kotyan, Danilo Vasconcellos Vargas

**Abstract**: Vision Transformer (ViT) has demonstrated promising performance in computer vision tasks, comparable to state-of-the-art neural networks. Yet, this new type of deep neural network architecture is vulnerable to adversarial attacks limiting its capabilities in terms of robustness. This article presents a novel contribution aimed at further improving the accuracy and robustness of ViT, particularly in the face of adversarial attacks. We propose an augmentation technique called `Dynamic Scanning Augmentation' that leverages dynamic input sequences to adaptively focus on different patches, thereby maintaining performance and robustness. Our detailed investigations reveal that this adaptability to the input sequence induces significant changes in the attention mechanism of ViT, even for the same image. We introduce four variations of Dynamic Scanning Augmentation, outperforming ViT in terms of both robustness to adversarial attacks and accuracy against natural images, with one variant showing comparable results. By integrating our augmentation technique, we observe a substantial increase in ViT's robustness, improving it from $17\%$ to $92\%$ measured across different types of adversarial attacks. These findings, together with other comprehensive tests, indicate that Dynamic Scanning Augmentation enhances accuracy and robustness by promoting a more adaptive type of attention. In conclusion, this work contributes to the ongoing research on Vision Transformers by introducing Dynamic Scanning Augmentation as a technique for improving the accuracy and robustness of ViT. The observed results highlight the potential of this approach in advancing computer vision tasks and merit further exploration in future studies.



## **3. NEO-KD: Knowledge-Distillation-Based Adversarial Training for Robust Multi-Exit Neural Networks**

cs.LG

10 pages, 4 figures, accepted by 37th Conference on Neural  Information Processing Systems (NeurIPS 2023)

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00428v1) [paper-pdf](http://arxiv.org/pdf/2311.00428v1)

**Authors**: Seokil Ham, Jungwuk Park, Dong-Jun Han, Jaekyun Moon

**Abstract**: While multi-exit neural networks are regarded as a promising solution for making efficient inference via early exits, combating adversarial attacks remains a challenging problem. In multi-exit networks, due to the high dependency among different submodels, an adversarial example targeting a specific exit not only degrades the performance of the target exit but also reduces the performance of all other exits concurrently. This makes multi-exit networks highly vulnerable to simple adversarial attacks. In this paper, we propose NEO-KD, a knowledge-distillation-based adversarial training strategy that tackles this fundamental challenge based on two key contributions. NEO-KD first resorts to neighbor knowledge distillation to guide the output of the adversarial examples to tend to the ensemble outputs of neighbor exits of clean data. NEO-KD also employs exit-wise orthogonal knowledge distillation for reducing adversarial transferability across different submodels. The result is a significantly improved robustness against adversarial attacks. Experimental results on various datasets/models show that our method achieves the best adversarial accuracy with reduced computation budgets, compared to the baselines relying on existing adversarial training or knowledge distillation techniques for multi-exit networks.



## **4. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

cs.CV

arXiv admin note: text overlap with arXiv:2112.09428

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2210.08159v3) [paper-pdf](http://arxiv.org/pdf/2210.08159v3)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods. Code is available at https://github.com/antao97/LGM.



## **5. LFAA: Crafting Transferable Targeted Adversarial Examples with Low-Frequency Perturbations**

cs.CV

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2310.20175v2) [paper-pdf](http://arxiv.org/pdf/2310.20175v2)

**Authors**: Kunyu Wang, Juluan Shi, Wenxuan Wang

**Abstract**: Deep neural networks are susceptible to adversarial attacks, which pose a significant threat to their security and reliability in real-world applications. The most notable adversarial attacks are transfer-based attacks, where an adversary crafts an adversarial example to fool one model, which can also fool other models. While previous research has made progress in improving the transferability of untargeted adversarial examples, the generation of targeted adversarial examples that can transfer between models remains a challenging task. In this work, we present a novel approach to generate transferable targeted adversarial examples by exploiting the vulnerability of deep neural networks to perturbations on high-frequency components of images. We observe that replacing the high-frequency component of an image with that of another image can mislead deep models, motivating us to craft perturbations containing high-frequency information to achieve targeted attacks. To this end, we propose a method called Low-Frequency Adversarial Attack (\name), which trains a conditional generator to generate targeted adversarial perturbations that are then added to the low-frequency component of the image. Extensive experiments on ImageNet demonstrate that our proposed approach significantly outperforms state-of-the-art methods, improving targeted attack success rates by a margin from 3.2\% to 15.5\%.



## **6. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

cs.CR

**SubmitDate**: 2023-11-01    [abs](http://arxiv.org/abs/2311.00207v1) [paper-pdf](http://arxiv.org/pdf/2311.00207v1)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer components, and wireless domain constraints. This paper proposes Magmaw, the first black-box attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on ML-based downstream applications. The resilience of the attack to the existing widely used defense methods of adversarial training and perturbation signal subtraction is experimentally verified. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of the defense mechanisms. Surprisingly, Magmaw is also effective against encrypted communication channels and conventional communications.



## **7. Robust Safety Classifier for Large Language Models: Adversarial Prompt Shield**

cs.CL

11 pages, 2 figures

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2311.00172v1) [paper-pdf](http://arxiv.org/pdf/2311.00172v1)

**Authors**: Jinhwa Kim, Ali Derakhshan, Ian G. Harris

**Abstract**: Large Language Models' safety remains a critical concern due to their vulnerability to adversarial attacks, which can prompt these systems to produce harmful responses. In the heart of these systems lies a safety classifier, a computational model trained to discern and mitigate potentially harmful, offensive, or unethical outputs. However, contemporary safety classifiers, despite their potential, often fail when exposed to inputs infused with adversarial noise. In response, our study introduces the Adversarial Prompt Shield (APS), a lightweight model that excels in detection accuracy and demonstrates resilience against adversarial prompts. Additionally, we propose novel strategies for autonomously generating adversarial training datasets, named Bot Adversarial Noisy Dialogue (BAND) datasets. These datasets are designed to fortify the safety classifier's robustness, and we investigate the consequences of incorporating adversarial examples into the training process. Through evaluations involving Large Language Models, we demonstrate that our classifier has the potential to decrease the attack success rate resulting from adversarial attacks by up to 60%. This advancement paves the way for the next generation of more reliable and resilient conversational agents.



## **8. Amoeba: Circumventing ML-supported Network Censorship via Adversarial Reinforcement Learning**

cs.CR

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20469v1) [paper-pdf](http://arxiv.org/pdf/2310.20469v1)

**Authors**: Haoyu Liu, Alec F. Diallo, Paul Patras

**Abstract**: Embedding covert streams into a cover channel is a common approach to circumventing Internet censorship, due to censors' inability to examine encrypted information in otherwise permitted protocols (Skype, HTTPS, etc.). However, recent advances in machine learning (ML) enable detecting a range of anti-censorship systems by learning distinct statistical patterns hidden in traffic flows. Therefore, designing obfuscation solutions able to generate traffic that is statistically similar to innocuous network activity, in order to deceive ML-based classifiers at line speed, is difficult.   In this paper, we formulate a practical adversarial attack strategy against flow classifiers as a method for circumventing censorship. Specifically, we cast the problem of finding adversarial flows that will be misclassified as a sequence generation task, which we solve with Amoeba, a novel reinforcement learning algorithm that we design. Amoeba works by interacting with censoring classifiers without any knowledge of their model structure, but by crafting packets and observing the classifiers' decisions, in order to guide the sequence generation process. Our experiments using data collected from two popular anti-censorship systems demonstrate that Amoeba can effectively shape adversarial flows that have on average 94% attack success rate against a range of ML algorithms. In addition, we show that these adversarial flows are robust in different network environments and possess transferability across various ML models, meaning that once trained against one, our agent can subvert other censoring classifiers without retraining.



## **9. On Extracting Specialized Code Abilities from Large Language Models: A Feasibility Study**

cs.SE

13 pages

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2303.03012v4) [paper-pdf](http://arxiv.org/pdf/2303.03012v4)

**Authors**: Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao, Yang Liu

**Abstract**: Recent advances in large language models (LLMs) significantly boost their usage in software engineering. However, training a well-performing LLM demands a substantial workforce for data collection and annotation. Moreover, training datasets may be proprietary or partially open, and the process often requires a costly GPU cluster. The intellectual property value of commercial LLMs makes them attractive targets for imitation attacks, but creating an imitation model with comparable parameters still incurs high costs. This motivates us to explore a practical and novel direction: slicing commercial black-box LLMs using medium-sized backbone models. In this paper, we explore the feasibility of launching imitation attacks on LLMs to extract their specialized code abilities, such as"code synthesis" and "code translation." We systematically investigate the effectiveness of launching code ability extraction attacks under different code-related tasks with multiple query schemes, including zero-shot, in-context, and Chain-of-Thought. We also design response checks to refine the outputs, leading to an effective imitation training process. Our results show promising outcomes, demonstrating that with a reasonable number of queries, attackers can train a medium-sized backbone model to replicate specialized code behaviors similar to the target LLMs. We summarize our findings and insights to help researchers better understand the threats posed by imitation attacks, including revealing a practical attack surface for generating adversarial code examples against LLMs.



## **10. Robust nonparametric regression based on deep ReLU neural networks**

stat.ME

40 pages

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20294v1) [paper-pdf](http://arxiv.org/pdf/2310.20294v1)

**Authors**: Juntong Chen

**Abstract**: In this paper, we consider robust nonparametric regression using deep neural networks with ReLU activation function. While several existing theoretically justified methods are geared towards robustness against identical heavy-tailed noise distributions, the rise of adversarial attacks has emphasized the importance of safeguarding estimation procedures against systematic contamination. We approach this statistical issue by shifting our focus towards estimating conditional distributions. To address it robustly, we introduce a novel estimation procedure based on $\ell$-estimation. Under a mild model assumption, we establish general non-asymptotic risk bounds for the resulting estimators, showcasing their robustness against contamination, outliers, and model misspecification. We then delve into the application of our approach using deep ReLU neural networks. When the model is well-specified and the regression function belongs to an $\alpha$-H\"older class, employing $\ell$-type estimation on suitable networks enables the resulting estimators to achieve the minimax optimal rate of convergence. Additionally, we demonstrate that deep $\ell$-type estimators can circumvent the curse of dimensionality by assuming the regression function closely resembles the composition of several H\"older functions. To attain this, new deep fully-connected ReLU neural networks have been designed to approximate this composition class. This approximation result can be of independent interest.



## **11. CFDP: Common Frequency Domain Pruning**

cs.CV

CVPR ECV 2023 Accepted Paper

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2306.04147v2) [paper-pdf](http://arxiv.org/pdf/2306.04147v2)

**Authors**: Samir Khaki, Weihan Luo

**Abstract**: As the saying goes, sometimes less is more -- and when it comes to neural networks, that couldn't be more true. Enter pruning, the art of selectively trimming away unnecessary parts of a network to create a more streamlined, efficient architecture. In this paper, we introduce a novel end-to-end pipeline for model pruning via the frequency domain. This work aims to shed light on the interoperability of intermediate model outputs and their significance beyond the spatial domain. Our method, dubbed Common Frequency Domain Pruning (CFDP) aims to extrapolate common frequency characteristics defined over the feature maps to rank the individual channels of a layer based on their level of importance in learning the representation. By harnessing the power of CFDP, we have achieved state-of-the-art results on CIFAR-10 with GoogLeNet reaching an accuracy of 95.25%, that is, +0.2% from the original model. We also outperform all benchmarks and match the original model's performance on ImageNet, using only 55% of the trainable parameters and 60% of the FLOPs. In addition to notable performances, models produced via CFDP exhibit robustness to a variety of configurations including pruning from untrained neural architectures, and resistance to adversarial attacks. The implementation code can be found at https://github.com/Skhaki18/CFDP.



## **12. BERT Lost Patience Won't Be Robust to Adversarial Slowdown**

cs.LG

Accepted to NeurIPS 2023 [Poster]

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.19152v2) [paper-pdf](http://arxiv.org/pdf/2310.19152v2)

**Authors**: Zachary Coalson, Gabriel Ritter, Rakesh Bobba, Sanghyun Hong

**Abstract**: In this paper, we systematically evaluate the robustness of multi-exit language models against adversarial slowdown. To audit their robustness, we design a slowdown attack that generates natural adversarial text bypassing early-exit points. We use the resulting WAFFLE attack as a vehicle to conduct a comprehensive evaluation of three multi-exit mechanisms with the GLUE benchmark against adversarial slowdown. We then show our attack significantly reduces the computational savings provided by the three methods in both white-box and black-box settings. The more complex a mechanism is, the more vulnerable it is to adversarial slowdown. We also perform a linguistic analysis of the perturbed text inputs, identifying common perturbation patterns that our attack generates, and comparing them with standard adversarial text attacks. Moreover, we show that adversarial training is ineffective in defeating our slowdown attack, but input sanitization with a conversational model, e.g., ChatGPT, can remove perturbations effectively. This result suggests that future work is needed for developing efficient yet robust multi-exit models. Our code is available at: https://github.com/ztcoalson/WAFFLE



## **13. Is Robustness Transferable across Languages in Multilingual Neural Machine Translation?**

cs.AI

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2310.20162v1) [paper-pdf](http://arxiv.org/pdf/2310.20162v1)

**Authors**: Leiyu Pan, Supryadi, Deyi Xiong

**Abstract**: Robustness, the ability of models to maintain performance in the face of perturbations, is critical for developing reliable NLP systems. Recent studies have shown promising results in improving the robustness of models through adversarial training and data augmentation. However, in machine translation, most of these studies have focused on bilingual machine translation with a single translation direction. In this paper, we investigate the transferability of robustness across different languages in multilingual neural machine translation. We propose a robustness transfer analysis protocol and conduct a series of experiments. In particular, we use character-, word-, and multi-level noises to attack the specific translation direction of the multilingual neural machine translation model and evaluate the robustness of other translation directions. Our findings demonstrate that the robustness gained in one translation direction can indeed transfer to other translation directions. Additionally, we empirically find scenarios where robustness to character-level noise and word-level noise is more likely to transfer.



## **14. TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models**

cs.CR

Accepted by NeurIPS'23

**SubmitDate**: 2023-10-31    [abs](http://arxiv.org/abs/2306.06815v3) [paper-pdf](http://arxiv.org/pdf/2306.06815v3)

**Authors**: Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Boloni, Qian Lou

**Abstract**: Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks. Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.



## **15. Exploring Geometry of Blind Spots in Vision Models**

cs.CV

25 pages, 20 figures, Accepted at NeurIPS 2023 (spotlight)

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19889v1) [paper-pdf](http://arxiv.org/pdf/2310.19889v1)

**Authors**: Sriram Balasubramanian, Gaurang Sriramanan, Vinu Sankar Sadasivan, Soheil Feizi

**Abstract**: Despite the remarkable success of deep neural networks in a myriad of settings, several works have demonstrated their overwhelming sensitivity to near-imperceptible perturbations, known as adversarial attacks. On the other hand, prior works have also observed that deep networks can be under-sensitive, wherein large-magnitude perturbations in input space do not induce appreciable changes to network activations. In this work, we study in detail the phenomenon of under-sensitivity in vision models such as CNNs and Transformers, and present techniques to study the geometry and extent of "equi-confidence" level sets of such networks. We propose a Level Set Traversal algorithm that iteratively explores regions of high confidence with respect to the input space using orthogonal components of the local gradients. Given a source image, we use this algorithm to identify inputs that lie in the same equi-confidence level set as the source image despite being perceptually similar to arbitrary images from other classes. We further observe that the source image is linearly connected by a high-confidence path to these inputs, uncovering a star-like structure for level sets of deep networks. Furthermore, we attempt to identify and estimate the extent of these connected higher-dimensional regions over which the model maintains a high degree of confidence. The code for this project is publicly available at https://github.com/SriramB-98/blindspots-neurips-sub



## **16. Adversarial Attacks and Defenses in Large Language Models: Old and New Threats**

cs.AI

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19737v1) [paper-pdf](http://arxiv.org/pdf/2310.19737v1)

**Authors**: Leo Schwinn, David Dobre, Stephan Günnemann, Gauthier Gidel

**Abstract**: Over the past decade, there has been extensive research aimed at enhancing the robustness of neural networks, yet this problem remains vastly unsolved. Here, one major impediment has been the overestimation of the robustness of new defense approaches due to faulty defense evaluations. Flawed robustness evaluations necessitate rectifications in subsequent works, dangerously slowing down the research and providing a false sense of security. In this context, we will face substantial challenges associated with an impending adversarial arms race in natural language processing, specifically with closed-source Large Language Models (LLMs), such as ChatGPT, Google Bard, or Anthropic's Claude. We provide a first set of prerequisites to improve the robustness assessment of new approaches and reduce the amount of faulty evaluations. Additionally, we identify embedding space attacks on LLMs as another viable threat model for the purposes of generating malicious content in open-sourced models. Finally, we demonstrate on a recently proposed defense that, without LLM-specific best practices in place, it is easy to overestimate the robustness of a new approach.



## **17. Differentially Private Reward Estimation with Preference Feedback**

cs.LG

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19733v1) [paper-pdf](http://arxiv.org/pdf/2310.19733v1)

**Authors**: Sayak Ray Chowdhury, Xingyu Zhou, Nagarajan Natarajan

**Abstract**: Learning from preference-based feedback has recently gained considerable traction as a promising approach to align generative models with human interests. Instead of relying on numerical rewards, the generative models are trained using reinforcement learning with human feedback (RLHF). These approaches first solicit feedback from human labelers typically in the form of pairwise comparisons between two possible actions, then estimate a reward model using these comparisons, and finally employ a policy based on the estimated reward model. An adversarial attack in any step of the above pipeline might reveal private and sensitive information of human labelers. In this work, we adopt the notion of label differential privacy (DP) and focus on the problem of reward estimation from preference-based feedback while protecting privacy of each individual labelers. Specifically, we consider the parametric Bradley-Terry-Luce (BTL) model for such pairwise comparison feedback involving a latent reward parameter $\theta^* \in \mathbb{R}^d$. Within a standard minimax estimation framework, we provide tight upper and lower bounds on the error in estimating $\theta^*$ under both local and central models of DP. We show, for a given privacy budget $\epsilon$ and number of samples $n$, that the additional cost to ensure label-DP under local model is $\Theta \big(\frac{1}{ e^\epsilon-1}\sqrt{\frac{d}{n}}\big)$, while it is $\Theta\big(\frac{\text{poly}(d)}{\epsilon n} \big)$ under the weaker central model. We perform simulations on synthetic data that corroborate these theoretical results.



## **18. Iris: Dynamic Privacy Preserving Search in Structured Peer-to-Peer Networks**

cs.CR

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19634v1) [paper-pdf](http://arxiv.org/pdf/2310.19634v1)

**Authors**: Angeliki Aktypi, Kasper Rasmussen

**Abstract**: In structured peer-to-peer networks like Chord, the users manage to retrieve the information they seek by asking other nodes from the network for the information they search. Revealing to other nodes the search target makes structured peer-to-peer networks unsuitable for applications that demand query privacy, i.e., hiding the query's target from the intermediate nodes that take part in the routing. This paper studies the query privacy of structured P2P networks, particularly the Chord protocol.   We initially observe that already proposed privacy notions, such as $k$-anonymity, do not allow us to reason about the privacy guarantees of a query in Chord in the presence of a strong adversary. Thus, we introduce a new privacy notion that we call $(\alpha,\delta)$-privacy that allows us to evaluate the privacy guarantees even when considering the worst-case scenario regarding an attacker's background knowledge.   We then design Iris, an algorithm that allows a requester to conceal the target of a query in Chord from the intermediate nodes that take part in the routing. Iris achieves that by having the requester query for other than the target addresses so as reaching each one of them allows the requester to get closer to the target address.   We perform a security analysis of the proposed algorithm, based on the privacy notion we introduce. We also develop a prototype of the algorithm in Matlab and evaluate its performance. Our analysis proves Iris to be $(\alpha,\delta)$-private while introducing a modest performance overhead.



## **19. Testing Robustness Against Unforeseen Adversaries**

cs.LG

Datasets available at  https://github.com/centerforaisafety/adversarial-corruptions

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/1908.08016v4) [paper-pdf](http://arxiv.org/pdf/1908.08016v4)

**Authors**: Max Kaufmann, Daniel Kang, Yi Sun, Steven Basart, Xuwang Yin, Mantas Mazeika, Akul Arora, Adam Dziedzic, Franziska Boenisch, Tom Brown, Jacob Steinhardt, Dan Hendrycks

**Abstract**: Adversarial robustness research primarily focuses on L_p perturbations, and most defenses are developed with identical training-time and test-time adversaries. However, in real-world applications developers are unlikely to have access to the full range of attacks or corruptions their system will face. Furthermore, worst-case inputs are likely to be diverse and need not be constrained to the L_p ball. To narrow in on this discrepancy between research and reality we introduce ImageNet-UA, a framework for evaluating model robustness against a range of unforeseen adversaries, including eighteen new non-L_p attacks. To perform well on ImageNet-UA, defenses must overcome a generalization gap and be robust to a diverse attacks not encountered during training. In extensive experiments, we find that existing robustness measures do not capture unforeseen robustness, that standard robustness techniques are beat by alternative training strategies, and that novel methods can improve unforeseen robustness. We present ImageNet-UA as a useful tool for the community for improving the worst-case behavior of machine learning systems.



## **20. A Generative Framework for Low-Cost Result Validation of Outsourced Machine Learning Tasks**

cs.CR

16 pages, 11 figures

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2304.00083v3) [paper-pdf](http://arxiv.org/pdf/2304.00083v3)

**Authors**: Abhinav Kumar, Miguel A. Guirao Aguilera, Reza Tourani, Satyajayant Misra

**Abstract**: The growing popularity of Machine Learning (ML) has led to its deployment in various sensitive domains, which has resulted in significant research focused on ML security and privacy. However, in some applications, such as autonomous driving, integrity verification of the outsourced ML workload is more critical--a facet that has not received much attention. Existing solutions, such as multi-party computation and proof-based systems, impose significant computation overhead, which makes them unfit for real-time applications. We propose Fides, a novel framework for real-time validation of outsourced ML workloads. Fides features a novel and efficient distillation technique--Greedy Distillation Transfer Learning--that dynamically distills and fine-tunes a space and compute-efficient verification model for verifying the corresponding service model while running inside a trusted execution environment. Fides features a client-side attack detection model that uses statistical analysis and divergence measurements to identify, with a high likelihood, if the service model is under attack. Fides also offers a re-classification functionality that predicts the original class whenever an attack is identified. We devised a generative adversarial network framework for training the attack detection and re-classification models. The evaluation shows that Fides achieves an accuracy of up to 98% for attack detection and 94% for re-classification.



## **21. Generated Distributions Are All You Need for Membership Inference Attacks Against Generative Models**

cs.CR

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19410v1) [paper-pdf](http://arxiv.org/pdf/2310.19410v1)

**Authors**: Minxing Zhang, Ning Yu, Rui Wen, Michael Backes, Yang Zhang

**Abstract**: Generative models have demonstrated revolutionary success in various visual creation tasks, but in the meantime, they have been exposed to the threat of leaking private information of their training data. Several membership inference attacks (MIAs) have been proposed to exhibit the privacy vulnerability of generative models by classifying a query image as a training dataset member or nonmember. However, these attacks suffer from major limitations, such as requiring shadow models and white-box access, and either ignoring or only focusing on the unique property of diffusion models, which block their generalization to multiple generative models. In contrast, we propose the first generalized membership inference attack against a variety of generative models such as generative adversarial networks, [variational] autoencoders, implicit functions, and the emerging diffusion models. We leverage only generated distributions from target generators and auxiliary non-member datasets, therefore regarding target generators as black boxes and agnostic to their architectures or application scenarios. Experiments validate that all the generative models are vulnerable to our attack. For instance, our work achieves attack AUC $>0.99$ against DDPM, DDIM, and FastDPM trained on CIFAR-10 and CelebA. And the attack against VQGAN, LDM (for the text-conditional generation), and LIIF achieves AUC $>0.90.$ As a result, we appeal to our community to be aware of such privacy leakage risks when designing and publishing generative models.



## **22. Balance, Imbalance, and Rebalance: Understanding Robust Overfitting from a Minimax Game Perspective**

cs.LG

Accepted by NeurIPS 2023

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19360v1) [paper-pdf](http://arxiv.org/pdf/2310.19360v1)

**Authors**: Yifei Wang, Liangchen Li, Jiansheng Yang, Zhouchen Lin, Yisen Wang

**Abstract**: Adversarial Training (AT) has become arguably the state-of-the-art algorithm for extracting robust features. However, researchers recently notice that AT suffers from severe robust overfitting problems, particularly after learning rate (LR) decay. In this paper, we explain this phenomenon by viewing adversarial training as a dynamic minimax game between the model trainer and the attacker. Specifically, we analyze how LR decay breaks the balance between the minimax game by empowering the trainer with a stronger memorization ability, and show such imbalance induces robust overfitting as a result of memorizing non-robust features. We validate this understanding with extensive experiments, and provide a holistic view of robust overfitting from the dynamics of both the two game players. This understanding further inspires us to alleviate robust overfitting by rebalancing the two players by either regularizing the trainer's capacity or improving the attack strength. Experiments show that the proposed ReBalanced Adversarial Training (ReBAT) can attain good robustness and does not suffer from robust overfitting even after very long training. Code is available at https://github.com/PKU-ML/ReBAT.



## **23. Label-Only Model Inversion Attacks via Knowledge Transfer**

cs.LG

Accepted to Neurips 2023. The first two authors contributed equally

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2310.19342v1) [paper-pdf](http://arxiv.org/pdf/2310.19342v1)

**Authors**: Ngoc-Bao Nguyen, Keshigeyan Chandrasegaran, Milad Abdollahzadeh, Ngai-Man Cheung

**Abstract**: In a model inversion (MI) attack, an adversary abuses access to a machine learning (ML) model to infer and reconstruct private training data. Remarkable progress has been made in the white-box and black-box setups, where the adversary has access to the complete model or the model's soft output respectively. However, there is very limited study in the most challenging but practically important setup: Label-only MI attacks, where the adversary only has access to the model's predicted label (hard label) without confidence scores nor any other model information.   In this work, we propose LOKT, a novel approach for label-only MI attacks. Our idea is based on transfer of knowledge from the opaque target model to surrogate models. Subsequently, using these surrogate models, our approach can harness advanced white-box attacks. We propose knowledge transfer based on generative modelling, and introduce a new model, Target model-assisted ACGAN (T-ACGAN), for effective knowledge transfer. Our method casts the challenging label-only MI into the more tractable white-box setup. We provide analysis to support that surrogate models based on our approach serve as effective proxies for the target model for MI. Our experiments show that our method significantly outperforms existing SOTA Label-only MI attack by more than 15% across all MI benchmarks. Furthermore, our method compares favorably in terms of query budget. Our study highlights rising privacy threats for ML models even when minimal information (i.e., hard labels) is exposed. Our study highlights rising privacy threats for ML models even when minimal information (i.e., hard labels) is exposed. Our code, demo, models and reconstructed data are available at our project page: https://ngoc-nguyen-0.github.io/lokt/



## **24. Treatment Learning Causal Transformer for Noisy Image Classification**

cs.CV

Accepted to IEEE WACV 2023. The first version was finished in May  2018

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2203.15529v2) [paper-pdf](http://arxiv.org/pdf/2203.15529v2)

**Authors**: Chao-Han Huck Yang, I-Te Danny Hung, Yi-Chieh Liu, Pin-Yu Chen

**Abstract**: Current top-notch deep learning (DL) based vision models are primarily based on exploring and exploiting the inherent correlations between training data samples and their associated labels. However, a known practical challenge is their degraded performance against "noisy" data, induced by different circumstances such as spurious correlations, irrelevant contexts, domain shift, and adversarial attacks. In this work, we incorporate this binary information of "existence of noise" as treatment into image classification tasks to improve prediction accuracy by jointly estimating their treatment effects. Motivated from causal variational inference, we propose a transformer-based architecture, Treatment Learning Causal Transformer (TLT), that uses a latent generative model to estimate robust feature representations from current observational input for noise image classification. Depending on the estimated noise level (modeled as a binary treatment factor), TLT assigns the corresponding inference network trained by the designed causal loss for prediction. We also create new noisy image datasets incorporating a wide range of noise factors (e.g., object masking, style transfer, and adversarial perturbation) for performance benchmarking. The superior performance of TLT in noisy image classification is further validated by several refutation evaluation metrics. As a by-product, TLT also improves visual salience methods for perceiving noisy images.



## **25. A Black-Box Approach to Post-Quantum Zero-Knowledge in Constant Rounds**

quant-ph

Fixed a minor technical issue (see Footnote 17 in page 21) and  improved the proof of Claim 4.5. (10/30/2023)

**SubmitDate**: 2023-10-30    [abs](http://arxiv.org/abs/2011.02670v4) [paper-pdf](http://arxiv.org/pdf/2011.02670v4)

**Authors**: Nai-Hui Chia, Kai-Min Chung, Takashi Yamakawa

**Abstract**: In a recent seminal work, Bitansky and Shmueli (STOC '20) gave the first construction of a constant round zero-knowledge argument for NP secure against quantum attacks. However, their construction has several drawbacks compared to the classical counterparts. Specifically, their construction only achieves computational soundness, requires strong assumptions of quantum hardness of learning with errors (QLWE assumption) and the existence of quantum fully homomorphic encryption (QFHE), and relies on non-black-box simulation. In this paper, we resolve these issues at the cost of weakening the notion of zero-knowledge to what is called $\epsilon$-zero-knowledge. Concretely, we construct the following protocols:   - We construct a constant round interactive proof for NP that satisfies statistical soundness and black-box $\epsilon$-zero-knowledge against quantum attacks assuming the existence of collapsing hash functions, which is a quantum counterpart of collision-resistant hash functions. Interestingly, this construction is just an adapted version of the classical protocol by Goldreich and Kahan (JoC '96) though the proof of $\epsilon$-zero-knowledge property against quantum adversaries requires novel ideas.   - We construct a constant round interactive argument for NP that satisfies computational soundness and black-box $\epsilon$-zero-knowledge against quantum attacks only assuming the existence of post-quantum one-way functions.   At the heart of our results is a new quantum rewinding technique that enables a simulator to extract a committed message of a malicious verifier while simulating verifier's internal state in an appropriate sense.



## **26. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

cs.CR

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19181v1) [paper-pdf](http://arxiv.org/pdf/2310.19181v1)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs - ChatGPT (GPT 3.5 Turbo), GPT 4, Claude and Bard to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing emails and websites that can convincingly imitate well-known brands, and also deploy a range of evasive tactics for the latter to elude detection mechanisms employed by anti-phishing systems. Notably, these attacks can be generated using unmodified, or "vanilla," versions of these LLMs, without requiring any prior adversarial exploits such as jailbreaking. As a countermeasure, we build a BERT based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content attaining an accuracy of 97\% for phishing website prompts, and 94\% for phishing email prompts.



## **27. Robustifying Language Models with Test-Time Adaptation**

cs.CL

8 Pages 2 Figures Submitted to ICLR Workshop

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19177v1) [paper-pdf](http://arxiv.org/pdf/2310.19177v1)

**Authors**: Noah Thomas McDermott, Junfeng Yang, Chengzhi Mao

**Abstract**: Large-scale language models achieved state-of-the-art performance over a number of language tasks. However, they fail on adversarial language examples, which are sentences optimized to fool the language models but with similar semantic meanings for humans. While prior work focuses on making the language model robust at training time, retraining for robustness is often unrealistic for large-scale foundation models. Instead, we propose to make the language models robust at test time. By dynamically adapting the input sentence with predictions from masked words, we show that we can reverse many language adversarial attacks. Since our approach does not require any training, it works for novel tasks at test time and can adapt to novel adversarial corruptions. Visualizations and empirical results on two popular sentence classification datasets demonstrate that our method can repair adversarial language attacks over 65% o



## **28. RAIFLE: Reconstruction Attacks on Interaction-based Federated Learning with Active Data Manipulation**

cs.CR

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19163v1) [paper-pdf](http://arxiv.org/pdf/2310.19163v1)

**Authors**: Dzung Pham, Shreyas Kulkarni, Amir Houmansadr

**Abstract**: Federated learning (FL) has recently emerged as a privacy-preserving approach for machine learning in domains that rely on user interactions, particularly recommender systems (RS) and online learning to rank (OLTR). While there has been substantial research on the privacy of traditional FL, little attention has been paid to studying the privacy properties of these interaction-based FL (IFL) systems. In this work, we show that IFL can introduce unique challenges concerning user privacy, particularly when the central server has knowledge and control over the items that users interact with. Specifically, we demonstrate the threat of reconstructing user interactions by presenting RAIFLE, a general optimization-based reconstruction attack framework customized for IFL. RAIFLE employs Active Data Manipulation (ADM), a novel attack technique unique to IFL, where the server actively manipulates the training features of the items to induce adversarial behaviors in the local FL updates. We show that RAIFLE is more impactful than existing FL privacy attacks in the IFL context, and describe how it can undermine privacy defenses like secure aggregation and private information retrieval. Based on our findings, we propose and discuss countermeasure guidelines to mitigate our attack in the context of federated RS/OLTR specifically and IFL more broadly.



## **29. Poisoning Retrieval Corpora by Injecting Adversarial Passages**

cs.CL

EMNLP 2023. Our code is available at  https://github.com/princeton-nlp/corpus-poisoning

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19156v1) [paper-pdf](http://arxiv.org/pdf/2310.19156v1)

**Authors**: Zexuan Zhong, Ziqing Huang, Alexander Wettig, Danqi Chen

**Abstract**: Dense retrievers have achieved state-of-the-art performance in various information retrieval tasks, but to what extent can they be safely deployed in real-world applications? In this work, we propose a novel attack for dense retrieval systems in which a malicious user generates a small number of adversarial passages by perturbing discrete tokens to maximize similarity with a provided set of training queries. When these adversarial passages are inserted into a large retrieval corpus, we show that this attack is highly effective in fooling these systems to retrieve them for queries that were not seen by the attacker. More surprisingly, these adversarial passages can directly generalize to out-of-domain queries and corpora with a high success attack rate -- for instance, we find that 50 generated passages optimized on Natural Questions can mislead >94% of questions posed in financial documents or online forums. We also benchmark and compare a range of state-of-the-art dense retrievers, both unsupervised and supervised. Although different systems exhibit varying levels of vulnerability, we show they can all be successfully attacked by injecting up to 500 passages, a small fraction compared to a retrieval corpus of millions of passages.



## **30. Alignment with human representations supports robust few-shot learning**

cs.LG

Spotlight at NeurIPS 2023

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2301.11990v3) [paper-pdf](http://arxiv.org/pdf/2301.11990v3)

**Authors**: Ilia Sucholutsky, Thomas L. Griffiths

**Abstract**: Should we care whether AI systems have representations of the world that are similar to those of humans? We provide an information-theoretic analysis that suggests that there should be a U-shaped relationship between the degree of representational alignment with humans and performance on few-shot learning tasks. We confirm this prediction empirically, finding such a relationship in an analysis of the performance of 491 computer vision models. We also show that highly-aligned models are more robust to both natural adversarial attacks and domain shifts. Our results suggest that human-alignment is often a sufficient, but not necessary, condition for models to make effective use of limited data, be robust, and generalize well.



## **31. Boosting Decision-Based Black-Box Adversarial Attack with Gradient Priors**

cs.LG

Accepted by IJCAI 2023

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.19038v1) [paper-pdf](http://arxiv.org/pdf/2310.19038v1)

**Authors**: Han Liu, Xingshuo Huang, Xiaotong Zhang, Qimai Li, Fenglong Ma, Wei Wang, Hongyang Chen, Hong Yu, Xianchao Zhang

**Abstract**: Decision-based methods have shown to be effective in black-box adversarial attacks, as they can obtain satisfactory performance and only require to access the final model prediction. Gradient estimation is a critical step in black-box adversarial attacks, as it will directly affect the query efficiency. Recent works have attempted to utilize gradient priors to facilitate score-based methods to obtain better results. However, these gradient priors still suffer from the edge gradient discrepancy issue and the successive iteration gradient direction issue, thus are difficult to simply extend to decision-based methods. In this paper, we propose a novel Decision-based Black-box Attack framework with Gradient Priors (DBA-GP), which seamlessly integrates the data-dependent gradient prior and time-dependent prior into the gradient estimation procedure. First, by leveraging the joint bilateral filter to deal with each random perturbation, DBA-GP can guarantee that the generated perturbations in edge locations are hardly smoothed, i.e., alleviating the edge gradient discrepancy, thus remaining the characteristics of the original image as much as possible. Second, by utilizing a new gradient updating strategy to automatically adjust the successive iteration gradient direction, DBA-GP can accelerate the convergence speed, thus improving the query efficiency. Extensive experiments have demonstrated that the proposed method outperforms other strong baselines significantly.



## **32. Attacks on Online Learners: a Teacher-Student Analysis**

stat.ML

19 pages, 10 figures

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2305.11132v2) [paper-pdf](http://arxiv.org/pdf/2305.11132v2)

**Authors**: Riccardo Giuseppe Margiotta, Sebastian Goldt, Guido Sanguinetti

**Abstract**: Machine learning models are famously vulnerable to adversarial attacks: small ad-hoc perturbations of the data that can catastrophically alter the model predictions. While a large literature has studied the case of test-time attacks on pre-trained models, the important case of attacks in an online learning setting has received little attention so far. In this work, we use a control-theoretical perspective to study the scenario where an attacker may perturb data labels to manipulate the learning dynamics of an online learner. We perform a theoretical analysis of the problem in a teacher-student setup, considering different attack strategies, and obtaining analytical results for the steady state of simple linear learners. These results enable us to prove that a discontinuous transition in the learner's accuracy occurs when the attack strength exceeds a critical threshold. We then study empirically attacks on learners with complex architectures using real data, confirming the insights of our theoretical analysis. Our findings show that greedy attacks can be extremely efficient, especially when data stream in small batches.



## **33. Blacksmith: Fast Adversarial Training of Vision Transformers via a Mixture of Single-step and Multi-step Methods**

cs.CV

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.18975v1) [paper-pdf](http://arxiv.org/pdf/2310.18975v1)

**Authors**: Mahdi Salmani, Alireza Dehghanpour Farashah, Mohammad Azizmalayeri, Mahdi Amiri, Navid Eslami, Mohammad Taghi Manzuri, Mohammad Hossein Rohban

**Abstract**: Despite the remarkable success achieved by deep learning algorithms in various domains, such as computer vision, they remain vulnerable to adversarial perturbations. Adversarial Training (AT) stands out as one of the most effective solutions to address this issue; however, single-step AT can lead to Catastrophic Overfitting (CO). This scenario occurs when the adversarially trained network suddenly loses robustness against multi-step attacks like Projected Gradient Descent (PGD). Although several approaches have been proposed to address this problem in Convolutional Neural Networks (CNNs), we found out that they do not perform well when applied to Vision Transformers (ViTs). In this paper, we propose Blacksmith, a novel training strategy to overcome the CO problem, specifically in ViTs. Our approach utilizes either of PGD-2 or Fast Gradient Sign Method (FGSM) randomly in a mini-batch during the adversarial training of the neural network. This will increase the diversity of our training attacks, which could potentially mitigate the CO issue. To manage the increased training time resulting from this combination, we craft the PGD-2 attack based on only the first half of the layers, while FGSM is applied end-to-end. Through our experiments, we demonstrate that our novel method effectively prevents CO, achieves PGD-2 level performance, and outperforms other existing techniques including N-FGSM, which is the state-of-the-art method in fast training for CNNs.



## **34. Label Poisoning is All You Need**

cs.LG

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.18933v1) [paper-pdf](http://arxiv.org/pdf/2310.18933v1)

**Authors**: Rishi D. Jha, Jonathan Hayase, Sewoong Oh

**Abstract**: In a backdoor attack, an adversary injects corrupted data into a model's training dataset in order to gain control over its predictions on images with a specific attacker-defined trigger. A typical corrupted training example requires altering both the image, by applying the trigger, and the label. Models trained on clean images, therefore, were considered safe from backdoor attacks. However, in some common machine learning scenarios, the training labels are provided by potentially malicious third-parties. This includes crowd-sourced annotation and knowledge distillation. We, hence, investigate a fundamental question: can we launch a successful backdoor attack by only corrupting labels? We introduce a novel approach to design label-only backdoor attacks, which we call FLIP, and demonstrate its strengths on three datasets (CIFAR-10, CIFAR-100, and Tiny-ImageNet) and four architectures (ResNet-32, ResNet-18, VGG-19, and Vision Transformer). With only 2% of CIFAR-10 labels corrupted, FLIP achieves a near-perfect attack success rate of 99.4% while suffering only a 1.8% drop in the clean test accuracy. Our approach builds upon the recent advances in trajectory matching, originally introduced for dataset distillation.



## **35. Reliable learning in challenging environments**

cs.LG

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2304.03370v2) [paper-pdf](http://arxiv.org/pdf/2304.03370v2)

**Authors**: Maria-Florina Balcan, Steve Hanneke, Rattana Pukdee, Dravyansh Sharma

**Abstract**: The problem of designing learners that provide guarantees that their predictions are provably correct is of increasing importance in machine learning. However, learning theoretic guarantees have only been considered in very specific settings. In this work, we consider the design and analysis of reliable learners in challenging test-time environments as encountered in modern machine learning problems: namely `adversarial' test-time attacks (in several variations) and `natural' distribution shifts. In this work, we provide a reliable learner with provably optimal guarantees in such settings. We discuss computationally feasible implementations of the learner and further show that our algorithm achieves strong positive performance guarantees on several natural examples: for example, linear separators under log-concave distributions or smooth boundary classifiers under smooth probability distributions.



## **36. Trust, but Verify: Robust Image Segmentation using Deep Learning**

cs.CV

5 Pages, 8 Figures, conference

**SubmitDate**: 2023-10-29    [abs](http://arxiv.org/abs/2310.16999v2) [paper-pdf](http://arxiv.org/pdf/2310.16999v2)

**Authors**: Fahim Ahmed Zaman, Xiaodong Wu, Weiyu Xu, Milan Sonka, Raghuraman Mudumbai

**Abstract**: We describe a method for verifying the output of a deep neural network for medical image segmentation that is robust to several classes of random as well as worst-case perturbations i.e. adversarial attacks. This method is based on a general approach recently developed by the authors called "Trust, but Verify" wherein an auxiliary verification network produces predictions about certain masked features in the input image using the segmentation as an input. A well-designed auxiliary network will produce high-quality predictions when the input segmentations are accurate, but will produce low-quality predictions when the segmentations are incorrect. Checking the predictions of such a network with the original image allows us to detect bad segmentations. However, to ensure the verification method is truly robust, we need a method for checking the quality of the predictions that does not itself rely on a black-box neural network. Indeed, we show that previous methods for segmentation evaluation that do use deep neural regression networks are vulnerable to false negatives i.e. can inaccurately label bad segmentations as good. We describe the design of a verification network that avoids such vulnerability and present results to demonstrate its robustness compared to previous methods.



## **37. On the Exploitability of Instruction Tuning**

cs.CR

NeurIPS 2023 camera-ready (21 pages, 10 figures)

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2306.17194v2) [paper-pdf](http://arxiv.org/pdf/2306.17194v2)

**Authors**: Manli Shu, Jiongxiao Wang, Chen Zhu, Jonas Geiping, Chaowei Xiao, Tom Goldstein

**Abstract**: Instruction tuning is an effective technique to align large language models (LLMs) with human intents. In this work, we investigate how an adversary can exploit instruction tuning by injecting specific instruction-following examples into the training data that intentionally changes the model's behavior. For example, an adversary can achieve content injection by injecting training examples that mention target content and eliciting such behavior from downstream models. To achieve this goal, we propose \textit{AutoPoison}, an automated data poisoning pipeline. It naturally and coherently incorporates versatile attack goals into poisoned data with the help of an oracle LLM. We showcase two example attacks: content injection and over-refusal attacks, each aiming to induce a specific exploitable behavior. We quantify and benchmark the strength and the stealthiness of our data poisoning scheme. Our results show that AutoPoison allows an adversary to change a model's behavior by poisoning only a small fraction of data while maintaining a high level of stealthiness in the poisoned examples. We hope our work sheds light on how data quality affects the behavior of instruction-tuned models and raises awareness of the importance of data quality for responsible deployments of LLMs. Code is available at \url{https://github.com/azshue/AutoPoison}.



## **38. Purify++: Improving Diffusion-Purification with Advanced Diffusion Models and Control of Randomness**

cs.LG

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18762v1) [paper-pdf](http://arxiv.org/pdf/2310.18762v1)

**Authors**: Boya Zhang, Weijian Luo, Zhihua Zhang

**Abstract**: Adversarial attacks can mislead neural network classifiers. The defense against adversarial attacks is important for AI safety. Adversarial purification is a family of approaches that defend adversarial attacks with suitable pre-processing. Diffusion models have been shown to be effective for adversarial purification. Despite their success, many aspects of diffusion purification still remain unexplored. In this paper, we investigate and improve upon three limiting designs of diffusion purification: the use of an improved diffusion model, advanced numerical simulation techniques, and optimal control of randomness. Based on our findings, we propose Purify++, a new diffusion purification algorithm that is now the state-of-the-art purification method against several adversarial attacks. Our work presents a systematic exploration of the limits of diffusion purification methods.



## **39. PAC-Bayesian Spectrally-Normalized Bounds for Adversarially Robust Generalization**

cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.06182v2) [paper-pdf](http://arxiv.org/pdf/2310.06182v2)

**Authors**: Jiancong Xiao, Ruoyu Sun, Zhi- Quan Luo

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. It is found empirically that adversarially robust generalization is crucial in establishing defense algorithms against adversarial attacks. Therefore, it is interesting to study the theoretical guarantee of robust generalization. This paper focuses on norm-based complexity, based on a PAC-Bayes approach (Neyshabur et al., 2017). The main challenge lies in extending the key ingredient, which is a weight perturbation bound in standard settings, to the robust settings. Existing attempts heavily rely on additional strong assumptions, leading to loose bounds. In this paper, we address this issue and provide a spectrally-normalized robust generalization bound for DNNs. Compared to existing bounds, our bound offers two significant advantages: Firstly, it does not depend on additional assumptions. Secondly, it is considerably tighter, aligning with the bounds of standard generalization. Therefore, our result provides a different perspective on understanding robust generalization: The mismatch terms between standard and robust generalization bounds shown in previous studies do not contribute to the poor robust generalization. Instead, these disparities solely due to mathematical issues. Finally, we extend the main result to adversarial robustness against general non-$\ell_p$ attacks and other neural network architectures.



## **40. Revisiting Adversarial Training for ImageNet: Architectures, Training and Generalization across Threat Models**

cs.CV

Accepted at NeurIPS 2023

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2303.01870v2) [paper-pdf](http://arxiv.org/pdf/2303.01870v2)

**Authors**: Naman D Singh, Francesco Croce, Matthias Hein

**Abstract**: While adversarial training has been extensively studied for ResNet architectures and low resolution datasets like CIFAR, much less is known for ImageNet. Given the recent debate about whether transformers are more robust than convnets, we revisit adversarial training on ImageNet comparing ViTs and ConvNeXts. Extensive experiments show that minor changes in architecture, most notably replacing PatchStem with ConvStem, and training scheme have a significant impact on the achieved robustness. These changes not only increase robustness in the seen $\ell_\infty$-threat model, but even more so improve generalization to unseen $\ell_1/\ell_2$-attacks. Our modified ConvNeXt, ConvNeXt + ConvStem, yields the most robust $\ell_\infty$-models across different ranges of model parameters and FLOPs, while our ViT + ConvStem yields the best generalization to unseen threat models.



## **41. Enhancing Adversarial Robustness via Score-Based Optimization**

cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2307.04333v3) [paper-pdf](http://arxiv.org/pdf/2307.04333v3)

**Authors**: Boya Zhang, Weijian Luo, Zhihua Zhang

**Abstract**: Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.



## **42. Training Socially Aligned Language Models on Simulated Social Interactions**

cs.CL

Code, data, and models can be downloaded via  https://github.com/agi-templar/Stable-Alignment

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2305.16960v3) [paper-pdf](http://arxiv.org/pdf/2305.16960v3)

**Authors**: Ruibo Liu, Ruixin Yang, Chenyan Jia, Ge Zhang, Denny Zhou, Andrew M. Dai, Diyi Yang, Soroush Vosoughi

**Abstract**: Social alignment in AI systems aims to ensure that these models behave according to established societal values. However, unlike humans, who derive consensus on value judgments through social interaction, current language models (LMs) are trained to rigidly replicate their training corpus in isolation, leading to subpar generalization in unfamiliar scenarios and vulnerability to adversarial attacks. This work presents a novel training paradigm that permits LMs to learn from simulated social interactions. In comparison to existing methodologies, our approach is considerably more scalable and efficient, demonstrating superior performance in alignment benchmarks and human evaluations. This paradigm shift in the training of LMs brings us a step closer to developing AI systems that can robustly and accurately reflect societal norms and values.



## **43. Setting the Trap: Capturing and Defeating Backdoors in Pretrained Language Models through Honeypots**

cs.LG

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18633v1) [paper-pdf](http://arxiv.org/pdf/2310.18633v1)

**Authors**: Ruixiang Tang, Jiayi Yuan, Yiming Li, Zirui Liu, Rui Chen, Xia Hu

**Abstract**: In the field of natural language processing, the prevalent approach involves fine-tuning pretrained language models (PLMs) using local samples. Recent research has exposed the susceptibility of PLMs to backdoor attacks, wherein the adversaries can embed malicious prediction behaviors by manipulating a few training samples. In this study, our objective is to develop a backdoor-resistant tuning procedure that yields a backdoor-free model, no matter whether the fine-tuning dataset contains poisoned samples. To this end, we propose and integrate a honeypot module into the original PLM, specifically designed to absorb backdoor information exclusively. Our design is motivated by the observation that lower-layer representations in PLMs carry sufficient backdoor features while carrying minimal information about the original tasks. Consequently, we can impose penalties on the information acquired by the honeypot module to inhibit backdoor creation during the fine-tuning process of the stem network. Comprehensive experiments conducted on benchmark datasets substantiate the effectiveness and robustness of our defensive strategy. Notably, these results indicate a substantial reduction in the attack success rate ranging from 10\% to 40\% when compared to prior state-of-the-art methods.



## **44. Where have you been? A Study of Privacy Risk for Point-of-Interest Recommendation**

cs.LG

26 pages

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18606v1) [paper-pdf](http://arxiv.org/pdf/2310.18606v1)

**Authors**: Kunlin Cai, Jinghuai Zhang, Will Shand, Zhiqing Hong, Guang Wang, Desheng Zhang, Jianfeng Chi, Yuan Tian

**Abstract**: As location-based services (LBS) have grown in popularity, the collection of human mobility data has become increasingly extensive to build machine learning (ML) models offering enhanced convenience to LBS users. However, the convenience comes with the risk of privacy leakage since this type of data might contain sensitive information related to user identities, such as home/work locations. Prior work focuses on protecting mobility data privacy during transmission or prior to release, lacking the privacy risk evaluation of mobility data-based ML models. To better understand and quantify the privacy leakage in mobility data-based ML models, we design a privacy attack suite containing data extraction and membership inference attacks tailored for point-of-interest (POI) recommendation models, one of the most widely used mobility data-based ML models. These attacks in our attack suite assume different adversary knowledge and aim to extract different types of sensitive information from mobility data, providing a holistic privacy risk assessment for POI recommendation models. Our experimental evaluation using two real-world mobility datasets demonstrates that current POI recommendation models are vulnerable to our attacks. We also present unique findings to understand what types of mobility data are more susceptible to privacy attacks. Finally, we evaluate defenses against these attacks and highlight future directions and challenges.



## **45. Large Language Models Are Better Adversaries: Exploring Generative Clean-Label Backdoor Attacks Against Text Classifiers**

cs.LG

Accepted at EMNLP 2023 Findings

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18603v1) [paper-pdf](http://arxiv.org/pdf/2310.18603v1)

**Authors**: Wencong You, Zayd Hammoudeh, Daniel Lowd

**Abstract**: Backdoor attacks manipulate model predictions by inserting innocuous triggers into training and test data. We focus on more realistic and more challenging clean-label attacks where the adversarial training examples are correctly labeled. Our attack, LLMBkd, leverages language models to automatically insert diverse style-based triggers into texts. We also propose a poison selection technique to improve the effectiveness of both LLMBkd as well as existing textual backdoor attacks. Lastly, we describe REACT, a baseline defense to mitigate backdoor attacks via antidote training examples. Our evaluations demonstrate LLMBkd's effectiveness and efficiency, where we consistently achieve high attack success rates across a wide range of styles with little effort and no model training.



## **46. Towards Good Practices in Evaluating Transfer Adversarial Attacks**

cs.CR

An extended version can be found at arXiv:2310.11850. Code and a list  of categorized attacks are available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2211.09565v3) [paper-pdf](http://arxiv.org/pdf/2211.09565v3)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes

**Abstract**: Transfer adversarial attacks raise critical security concerns in real-world, black-box scenarios. However, the actual progress of this field is difficult to assess due to two common limitations in existing evaluations. First, different methods are often not systematically and fairly evaluated in a one-to-one comparison. Second, only transferability is evaluated but another key attack property, stealthiness, is largely overlooked. In this work, we design good practices to address these limitations, and we present the first comprehensive evaluation of transfer attacks, covering 23 representative attacks against 9 defenses on ImageNet. In particular, we propose to categorize existing attacks into five categories, which enables our systematic category-wise analyses. These analyses lead to new findings that even challenge existing knowledge and also help determine the optimal attack hyperparameters for our attack-wise comprehensive evaluation. We also pay particular attention to stealthiness, by adopting diverse imperceptibility metrics and looking into new, finer-grained characteristics. Overall, our new insights into transferability and stealthiness lead to actionable good practices for future evaluations.



## **47. A General Framework for Robust G-Invariance in G-Equivariant Networks**

cs.LG

**SubmitDate**: 2023-10-28    [abs](http://arxiv.org/abs/2310.18564v1) [paper-pdf](http://arxiv.org/pdf/2310.18564v1)

**Authors**: Sophia Sanborn, Nina Miolane

**Abstract**: We introduce a general method for achieving robust group-invariance in group-equivariant convolutional neural networks ($G$-CNNs), which we call the $G$-triple-correlation ($G$-TC) layer. The approach leverages the theory of the triple-correlation on groups, which is the unique, lowest-degree polynomial invariant map that is also complete. Many commonly used invariant maps - such as the max - are incomplete: they remove both group and signal structure. A complete invariant, by contrast, removes only the variation due to the actions of the group, while preserving all information about the structure of the signal. The completeness of the triple correlation endows the $G$-TC layer with strong robustness, which can be observed in its resistance to invariance-based adversarial attacks. In addition, we observe that it yields measurable improvements in classification accuracy over standard Max $G$-Pooling in $G$-CNN architectures. We provide a general and efficient implementation of the method for any discretized group, which requires only a table defining the group's product structure. We demonstrate the benefits of this method for $G$-CNNs defined on both commutative and non-commutative groups - $SO(2)$, $O(2)$, $SO(3)$, and $O(3)$ (discretized as the cyclic $C8$, dihedral $D16$, chiral octahedral $O$ and full octahedral $O_h$ groups) - acting on $\mathbb{R}^2$ and $\mathbb{R}^3$ on both $G$-MNIST and $G$-ModelNet10 datasets.



## **48. Understanding and Improving Ensemble Adversarial Defense**

cs.LG

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18477v1) [paper-pdf](http://arxiv.org/pdf/2310.18477v1)

**Authors**: Yian Deng, Tingting Mu

**Abstract**: The strategy of ensemble has become popular in adversarial defense, which trains multiple base classifiers to defend against adversarial attacks in a cooperative manner. Despite the empirical success, theoretical explanations on why an ensemble of adversarially trained classifiers is more robust than single ones remain unclear. To fill in this gap, we develop a new error theory dedicated to understanding ensemble adversarial defense, demonstrating a provable 0-1 loss reduction on challenging sample sets in an adversarial defense scenario. Guided by this theory, we propose an effective approach to improve ensemble adversarial defense, named interactive global adversarial training (iGAT). The proposal includes (1) a probabilistic distributing rule that selectively allocates to different base classifiers adversarial examples that are globally challenging to the ensemble, and (2) a regularization term to rescue the severest weaknesses of the base classifiers. Being tested over various existing ensemble adversarial defense techniques, iGAT is capable of boosting their performance by increases up to 17% evaluated using CIFAR10 and CIFAR100 datasets under both white-box and black-box attacks.



## **49. LipSim: A Provably Robust Perceptual Similarity Metric**

cs.CV

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18274v1) [paper-pdf](http://arxiv.org/pdf/2310.18274v1)

**Authors**: Sara Ghazanfari, Alexandre Araujo, Prashanth Krishnamurthy, Farshad Khorrami, Siddharth Garg

**Abstract**: Recent years have seen growing interest in developing and applying perceptual similarity metrics. Research has shown the superiority of perceptual metrics over pixel-wise metrics in aligning with human perception and serving as a proxy for the human visual system. On the other hand, as perceptual metrics rely on neural networks, there is a growing concern regarding their resilience, given the established vulnerability of neural networks to adversarial attacks. It is indeed logical to infer that perceptual metrics may inherit both the strengths and shortcomings of neural networks. In this work, we demonstrate the vulnerability of state-of-the-art perceptual similarity metrics based on an ensemble of ViT-based feature extractors to adversarial attacks. We then propose a framework to train a robust perceptual similarity metric called LipSim (Lipschitz Similarity Metric) with provable guarantees. By leveraging 1-Lipschitz neural networks as the backbone, LipSim provides guarded areas around each data point and certificates for all perturbations within an $\ell_2$ ball. Finally, a comprehensive set of experiments shows the performance of LipSim in terms of natural and certified scores and on the image retrieval application. The code is available at https://github.com/SaraGhazanfari/LipSim.



## **50. $α$-Mutual Information: A Tunable Privacy Measure for Privacy Protection in Data Sharing**

cs.LG

2023 22nd IEEE International Conference on Machine Learning and  Applications (ICMLA)

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18241v1) [paper-pdf](http://arxiv.org/pdf/2310.18241v1)

**Authors**: MirHamed Jafarzadeh Asl, Mohammadhadi Shateri, Fabrice Labeau

**Abstract**: This paper adopts Arimoto's $\alpha$-Mutual Information as a tunable privacy measure, in a privacy-preserving data release setting that aims to prevent disclosing private data to adversaries. By fine-tuning the privacy metric, we demonstrate that our approach yields superior models that effectively thwart attackers across various performance dimensions. We formulate a general distortion-based mechanism that manipulates the original data to offer privacy protection. The distortion metrics are determined according to the data structure of a specific experiment. We confront the problem expressed in the formulation by employing a general adversarial deep learning framework that consists of a releaser and an adversary, trained with opposite goals. This study conducts empirical experiments on images and time-series data to verify the functionality of $\alpha$-Mutual Information. We evaluate the privacy-utility trade-off of customized models and compare them to mutual information as the baseline measure. Finally, we analyze the consequence of an attacker's access to side information about private data and witness that adapting the privacy measure results in a more refined model than the state-of-the-art in terms of resiliency against side information.



