# Latest Adversarial Attack Papers
**update at 2023-07-17 15:31:14**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Bankrupting DoS Attackers**

cs.CR

**SubmitDate**: 2023-07-14    [abs](http://arxiv.org/abs/2205.08287v3) [paper-pdf](http://arxiv.org/pdf/2205.08287v3)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstract**: To defend against denial-of-service (DoS) attacks, we employ a technique called resource burning (RB). RB is the verifiable expenditure of a resource, such as computational power, required from clients before receiving service from the server. To the best of our knowledge, we present the first DoS defense algorithms where the algorithmic cost -- the cost to both the server and the honest clients -- is bounded as a function of the attacker's cost.   We model an omniscient, Byzantine attacker, and a server with access to an estimator that estimates the number of jobs from honest clients in any time interval. We examine two communication models: an idealized zero-latency model and a partially synchronous model. Notably, our algorithms for both models have asymptotically lower costs than the attacker's, as the attacker's costs grow large. Both algorithms use a simple rule to set required RB fees per job. We assume no prior knowledge of the number of jobs, the adversary's costs, or even the estimator's accuracy. However, these quantities parameterize the algorithms' costs.   We also prove a lower bound on the cost of any randomized algorithm. This lower bound shows that our algorithms achieve asymptotically tight costs as the number of jobs grows unbounded, whenever the estimator output is accurate to within a constant factor.



## **2. Boosting Backdoor Attack with A Learnable Poisoning Sample Selection Strategy**

cs.CR

**SubmitDate**: 2023-07-14    [abs](http://arxiv.org/abs/2307.07328v1) [paper-pdf](http://arxiv.org/pdf/2307.07328v1)

**Authors**: Zihao Zhu, Mingda Zhang, Shaokui Wei, Li Shen, Yanbo Fan, Baoyuan Wu

**Abstract**: Data-poisoning based backdoor attacks aim to insert backdoor into models by manipulating training datasets without controlling the training process of the target model. Existing attack methods mainly focus on designing triggers or fusion strategies between triggers and benign samples. However, they often randomly select samples to be poisoned, disregarding the varying importance of each poisoning sample in terms of backdoor injection. A recent selection strategy filters a fixed-size poisoning sample pool by recording forgetting events, but it fails to consider the remaining samples outside the pool from a global perspective. Moreover, computing forgetting events requires significant additional computing resources. Therefore, how to efficiently and effectively select poisoning samples from the entire dataset is an urgent problem in backdoor attacks.To address it, firstly, we introduce a poisoning mask into the regular backdoor training loss. We suppose that a backdoored model training with hard poisoning samples has a more backdoor effect on easy ones, which can be implemented by hindering the normal training process (\ie, maximizing loss \wrt mask). To further integrate it with normal training process, we then propose a learnable poisoning sample selection strategy to learn the mask together with the model parameters through a min-max optimization.Specifically, the outer loop aims to achieve the backdoor attack goal by minimizing the loss based on the selected samples, while the inner loop selects hard poisoning samples that impede this goal by maximizing the loss. After several rounds of adversarial training, we finally select effective poisoning samples with high contribution. Extensive experiments on benchmark datasets demonstrate the effectiveness and efficiency of our approach in boosting backdoor attack performance.



## **3. Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation**

eess.IV

This paper has been accepted in MICCAI 2023 conference

**SubmitDate**: 2023-07-14    [abs](http://arxiv.org/abs/2307.07269v1) [paper-pdf](http://arxiv.org/pdf/2307.07269v1)

**Authors**: Asif Hanif, Muzammal Naseer, Salman Khan, Mubarak Shah, Fahad Shahbaz Khan

**Abstract**: It is imperative to ensure the robustness of deep learning models in critical applications such as, healthcare. While recent advances in deep learning have improved the performance of volumetric medical image segmentation models, these models cannot be deployed for real-world applications immediately due to their vulnerability to adversarial attacks. We present a 3D frequency domain adversarial attack for volumetric medical image segmentation models and demonstrate its advantages over conventional input or voxel domain attacks. Using our proposed attack, we introduce a novel frequency domain adversarial training approach for optimizing a robust model against voxel and frequency domain attacks. Moreover, we propose frequency consistency loss to regulate our frequency domain adversarial training that achieves a better tradeoff between model's performance on clean and adversarial samples. Code is publicly available at https://github.com/asif-hanif/vafa.



## **4. Vulnerability-Aware Instance Reweighting For Adversarial Training**

cs.LG

**SubmitDate**: 2023-07-14    [abs](http://arxiv.org/abs/2307.07167v1) [paper-pdf](http://arxiv.org/pdf/2307.07167v1)

**Authors**: Olukorede Fakorede, Ashutosh Kumar Nirala, Modeste Atsague, Jin Tian

**Abstract**: Adversarial Training (AT) has been found to substantially improve the robustness of deep learning classifiers against adversarial attacks. AT involves obtaining robustness by including adversarial examples in training a classifier. Most variants of AT algorithms treat every training example equally. However, recent works have shown that better performance is achievable by treating them unequally. In addition, it has been observed that AT exerts an uneven influence on different classes in a training set and unfairly hurts examples corresponding to classes that are inherently harder to classify. Consequently, various reweighting schemes have been proposed that assign unequal weights to robust losses of individual examples in a training set. In this work, we propose a novel instance-wise reweighting scheme. It considers the vulnerability of each natural example and the resulting information loss on its adversarial counterpart occasioned by adversarial attacks. Through extensive experiments, we show that our proposed method significantly improves over existing reweighting schemes, especially against strong white and black-box attacks.



## **5. Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation**

cs.CL

Version 2; Work in progress

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2306.12916v2) [paper-pdf](http://arxiv.org/pdf/2306.12916v2)

**Authors**: Ran Zhang, Jihed Ouni, Steffen Eger

**Abstract**: While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We build the first CLCTS corpus, leveraging historical fictive texts and Wikipedia summaries in English and German, and examine the effectiveness of popular transformer end-to-end models with different intermediate finetuning tasks. Additionally, we explore the potential of ChatGPT for CLCTS as a summarizer and an evaluator. Overall, we report evaluations from humans, ChatGPT, and several recent automatic evaluation metrics where we find that our intermediate task finetuned end-to-end models generate bad to moderate quality summaries; ChatGPT as a summarizer (without any finetuning) provides moderate to good quality outputs and as an evaluator correlates moderately with human evaluations but is prone to giving lower scores. ChatGPT also seems very adept at normalizing historical text and outperforms context-unaware spelling normalization tools such as Norma. We finally test ChatGPT in a scenario with adversarially attacked and unseen source documents and find that ChatGPT profits from its prior knowledge to a certain degree, with better performances for omission and entity swap than negation against its prior knowledge. This benefit inflates its assessed quality as ChatGPT performs slightly worse for unseen source documents compared to seen documents. We additionally introspect our models' performances to find that longer, older and more complex source texts (all of which are more characteristic for historical language variants) are harder to summarize for all models, indicating the difficulty of the CLCTS task.



## **6. Visually Adversarial Attacks and Defenses in the Physical World: A Survey**

cs.CV

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2211.01671v5) [paper-pdf](http://arxiv.org/pdf/2211.01671v5)

**Authors**: Xingxing Wei, Bangzheng Pu, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they are vulnerable to adversarial examples. The current adversarial attacks in computer vision can be divided into digital attacks and physical attacks according to their different attack forms. Compared with digital attacks, which generate perturbations in the digital pixels, physical attacks are more practical in the real world. Owing to the serious security problem caused by physically adversarial examples, many works have been proposed to evaluate the physically adversarial robustness of DNNs in the past years. In this paper, we summarize a survey versus the current physically adversarial attacks and physically adversarial defenses in computer vision. To establish a taxonomy, we organize the current physical attacks from attack tasks, attack forms, and attack methods, respectively. Thus, readers can have a systematic knowledge of this topic from different aspects. For the physical defenses, we establish the taxonomy from pre-processing, in-processing, and post-processing for the DNN models to achieve full coverage of the adversarial defenses. Based on the above survey, we finally discuss the challenges of this research field and further outlook on the future direction.



## **7. Introducing Foundation Models as Surrogate Models: Advancing Towards More Practical Adversarial Attacks**

cs.LG

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.06608v1) [paper-pdf](http://arxiv.org/pdf/2307.06608v1)

**Authors**: Jiaming Zhang, Jitao Sang, Qi Yi

**Abstract**: Recently, the no-box adversarial attack, in which the attacker lacks access to the model's architecture, weights, and training data, become the most practical and challenging attack setup. However, there is an unawareness of the potential and flexibility inherent in the surrogate model selection process on no-box setting. Inspired by the burgeoning interest in utilizing foundational models to address downstream tasks, this paper adopts an innovative idea that 1) recasting adversarial attack as a downstream task. Specifically, image noise generation to meet the emerging trend and 2) introducing foundational models as surrogate models. Harnessing the concept of non-robust features, we elaborate on two guiding principles for surrogate model selection to explain why the foundational model is an optimal choice for this role. However, paradoxically, we observe that these foundational models underperform. Analyzing this unexpected behavior within the feature space, we attribute the lackluster performance of foundational models (e.g., CLIP) to their significant representational capacity and, conversely, their lack of discriminative prowess. To mitigate this issue, we propose the use of a margin-based loss strategy for the fine-tuning of foundational models on target images. The experimental results verify that our approach, which employs the basic Fast Gradient Sign Method (FGSM) attack algorithm, outstrips the performance of other, more convoluted algorithms. We conclude by advocating for the research community to consider surrogate models as crucial determinants in the effectiveness of adversarial attacks in no-box settings. The implications of our work bear relevance for improving the efficacy of such adversarial attacks and the overall robustness of AI systems.



## **8. Adversarial Policies Beat Superhuman Go AIs**

cs.LG

Accepted to ICML 2023, see paper for changelog

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2211.00241v4) [paper-pdf](http://arxiv.org/pdf/2211.00241v4)

**Authors**: Tony T. Wang, Adam Gleave, Tom Tseng, Kellin Pelrine, Nora Belrose, Joseph Miller, Michael D. Dennis, Yawen Duan, Viktor Pogrebniak, Sergey Levine, Stuart Russell

**Abstract**: We attack the state-of-the-art Go-playing AI system KataGo by training adversarial policies against it, achieving a >97% win rate against KataGo running at superhuman settings. Our adversaries do not win by playing Go well. Instead, they trick KataGo into making serious blunders. Our attack transfers zero-shot to other superhuman Go-playing AIs, and is comprehensible to the extent that human experts can implement it without algorithmic assistance to consistently beat superhuman AIs. The core vulnerability uncovered by our attack persists even in KataGo agents adversarially trained to defend against our attack. Our results demonstrate that even superhuman AI systems may harbor surprising failure modes. Example games are available https://goattack.far.ai/.



## **9. Multi-objective Evolutionary Search of Variable-length Composite Semantic Perturbations**

cs.CV

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.06548v1) [paper-pdf](http://arxiv.org/pdf/2307.06548v1)

**Authors**: Jialiang Suna, Wen Yao, Tingsong Jianga, Xiaoqian Chena

**Abstract**: Deep neural networks have proven to be vulnerable to adversarial attacks in the form of adding specific perturbations on images to make wrong outputs. Designing stronger adversarial attack methods can help more reliably evaluate the robustness of DNN models. To release the harbor burden and improve the attack performance, auto machine learning (AutoML) has recently emerged as one successful technique to help automatically find the near-optimal adversarial attack strategy. However, existing works about AutoML for adversarial attacks only focus on $L_{\infty}$-norm-based perturbations. In fact, semantic perturbations attract increasing attention due to their naturalnesses and physical realizability. To bridge the gap between AutoML and semantic adversarial attacks, we propose a novel method called multi-objective evolutionary search of variable-length composite semantic perturbations (MES-VCSP). Specifically, we construct the mathematical model of variable-length composite semantic perturbations, which provides five gradient-based semantic attack methods. The same type of perturbation in an attack sequence is allowed to be performed multiple times. Besides, we introduce the multi-objective evolutionary search consisting of NSGA-II and neighborhood search to find near-optimal variable-length attack sequences. Experimental results on CIFAR10 and ImageNet datasets show that compared with existing methods, MES-VCSP can obtain adversarial examples with a higher attack success rate, more naturalness, and less time cost.



## **10. Hiding in Plain Sight: Differential Privacy Noise Exploitation for Evasion-resilient Localized Poisoning Attacks in Multiagent Reinforcement Learning**

cs.LG

6 pages, 4 figures, Published in the proceeding of the ICMLC 2023,  9-11 July 2023, The University of Adelaide, Adelaide, Australia

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.00268v2) [paper-pdf](http://arxiv.org/pdf/2307.00268v2)

**Authors**: Md Tamjid Hossain, Hung La

**Abstract**: Lately, differential privacy (DP) has been introduced in cooperative multiagent reinforcement learning (CMARL) to safeguard the agents' privacy against adversarial inference during knowledge sharing. Nevertheless, we argue that the noise introduced by DP mechanisms may inadvertently give rise to a novel poisoning threat, specifically in the context of private knowledge sharing during CMARL, which remains unexplored in the literature. To address this shortcoming, we present an adaptive, privacy-exploiting, and evasion-resilient localized poisoning attack (PeLPA) that capitalizes on the inherent DP-noise to circumvent anomaly detection systems and hinder the optimal convergence of the CMARL model. We rigorously evaluate our proposed PeLPA attack in diverse environments, encompassing both non-adversarial and multiple-adversarial contexts. Our findings reveal that, in a medium-scale environment, the PeLPA attack with attacker ratios of 20% and 40% can lead to an increase in average steps to goal by 50.69% and 64.41%, respectively. Furthermore, under similar conditions, PeLPA can result in a 1.4x and 1.6x computational time increase in optimal reward attainment and a 1.18x and 1.38x slower convergence for attacker ratios of 20% and 40%, respectively.



## **11. The Butterfly Effect in AI Fairness and Bias**

cs.CY

Working Draft

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.05842v2) [paper-pdf](http://arxiv.org/pdf/2307.05842v2)

**Authors**: Emilio Ferrara

**Abstract**: The Butterfly Effect, a concept originating from chaos theory, underscores how small changes can have significant and unpredictable impacts on complex systems. In the context of AI fairness and bias, the Butterfly Effect can stem from a variety of sources, such as small biases or skewed data inputs during algorithm development, saddle points in training, or distribution shifts in data between training and testing phases. These seemingly minor alterations can lead to unexpected and substantial unfair outcomes, disproportionately affecting underrepresented individuals or groups and perpetuating pre-existing inequalities. Moreover, the Butterfly Effect can amplify inherent biases within data or algorithms, exacerbate feedback loops, and create vulnerabilities for adversarial attacks. Given the intricate nature of AI systems and their societal implications, it is crucial to thoroughly examine any changes to algorithms or input data for potential unintended consequences. In this paper, we envision both algorithmic and empirical strategies to detect, quantify, and mitigate the Butterfly Effect in AI systems, emphasizing the importance of addressing these challenges to promote fairness and ensure responsible AI development.



## **12. Microbial Genetic Algorithm-based Black-box Attack against Interpretable Deep Learning Systems**

cs.CV

**SubmitDate**: 2023-07-13    [abs](http://arxiv.org/abs/2307.06496v1) [paper-pdf](http://arxiv.org/pdf/2307.06496v1)

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, Simon S. Woo, Eric Chan-Tin, Tamer Abuhmed

**Abstract**: Deep learning models are susceptible to adversarial samples in white and black-box environments. Although previous studies have shown high attack success rates, coupling DNN models with interpretation models could offer a sense of security when a human expert is involved, who can identify whether a given sample is benign or malicious. However, in white-box environments, interpretable deep learning systems (IDLSes) have been shown to be vulnerable to malicious manipulations. In black-box settings, as access to the components of IDLSes is limited, it becomes more challenging for the adversary to fool the system. In this work, we propose a Query-efficient Score-based black-box attack against IDLSes, QuScore, which requires no knowledge of the target model and its coupled interpretation model. QuScore is based on transfer-based and score-based methods by employing an effective microbial genetic algorithm. Our method is designed to reduce the number of queries necessary to carry out successful attacks, resulting in a more efficient process. By continuously refining the adversarial samples created based on feedback scores from the IDLS, our approach effectively navigates the search space to identify perturbations that can fool the system. We evaluate the attack's effectiveness on four CNN models (Inception, ResNet, VGG, DenseNet) and two interpretation models (CAM, Grad), using both ImageNet and CIFAR datasets. Our results show that the proposed approach is query-efficient with a high attack success rate that can reach between 95% and 100% and transferability with an average success rate of 69% in the ImageNet and CIFAR datasets. Our attack method generates adversarial examples with attribution maps that resemble benign samples. We have also demonstrated that our attack is resilient against various preprocessing defense techniques and can easily be transferred to different DNN models.



## **13. Single-Class Target-Specific Attack against Interpretable Deep Learning Systems**

cs.CV

13 pages

**SubmitDate**: 2023-07-12    [abs](http://arxiv.org/abs/2307.06484v1) [paper-pdf](http://arxiv.org/pdf/2307.06484v1)

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, George K. Thiruvathukal, Hyoungshick Kim, Tamer Abuhmed

**Abstract**: In this paper, we present a novel Single-class target-specific Adversarial attack called SingleADV. The goal of SingleADV is to generate a universal perturbation that deceives the target model into confusing a specific category of objects with a target category while ensuring highly relevant and accurate interpretations. The universal perturbation is stochastically and iteratively optimized by minimizing the adversarial loss that is designed to consider both the classifier and interpreter costs in targeted and non-targeted categories. In this optimization framework, ruled by the first- and second-moment estimations, the desired loss surface promotes high confidence and interpretation score of adversarial samples. By avoiding unintended misclassification of samples from other categories, SingleADV enables more effective targeted attacks on interpretable deep learning systems in both white-box and black-box scenarios. To evaluate the effectiveness of SingleADV, we conduct experiments using four different model architectures (ResNet-50, VGG-16, DenseNet-169, and Inception-V3) coupled with three interpretation models (CAM, Grad, and MASK). Through extensive empirical evaluation, we demonstrate that SingleADV effectively deceives the target deep learning models and their associated interpreters under various conditions and settings. Our experimental results show that the performance of SingleADV is effective, with an average fooling ratio of 0.74 and an adversarial confidence level of 0.78 in generating deceptive adversarial samples. Furthermore, we discuss several countermeasures against SingleADV, including a transfer-based learning approach and existing preprocessing defenses.



## **14. Rational Neural Network Controllers**

eess.SY

20 Pages, 12 Figures

**SubmitDate**: 2023-07-12    [abs](http://arxiv.org/abs/2307.06287v1) [paper-pdf](http://arxiv.org/pdf/2307.06287v1)

**Authors**: Matthew Newton, Antonis Papachristodoulou

**Abstract**: Neural networks have shown great success in many machine learning related tasks, due to their ability to act as general function approximators. Recent work has demonstrated the effectiveness of neural networks in control systems (known as neural feedback loops), most notably by using a neural network as a controller. However, one of the big challenges of this approach is that neural networks have been shown to be sensitive to adversarial attacks. This means that, unless they are designed properly, they are not an ideal candidate for controllers due to issues with robustness and uncertainty, which are pivotal aspects of control systems. There has been initial work on robustness to both analyse and design dynamical systems with neural network controllers. However, one prominent issue with these methods is that they use existing neural network architectures tailored for traditional machine learning tasks. These structures may not be appropriate for neural network controllers and it is important to consider alternative architectures. This paper considers rational neural networks and presents novel rational activation functions, which can be used effectively in robustness problems for neural feedback loops. Rational activation functions are replaced by a general rational neural network structure, which is convex in the neural network's parameters. A method is proposed to recover a stabilising controller from a Sum of Squares feasibility test. This approach is then applied to a refined rational neural network which is more compatible with Sum of Squares programming. Numerical examples show that this method can successfully recover stabilising rational neural network controllers for neural feedback loops with non-linear plants with noise and parametric uncertainty.



## **15. I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models**

cs.CV

**SubmitDate**: 2023-07-12    [abs](http://arxiv.org/abs/2306.07591v2) [paper-pdf](http://arxiv.org/pdf/2306.07591v2)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Modern image-to-text systems typically adopt the encoder-decoder framework, which comprises two main components: an image encoder, responsible for extracting image features, and a transformer-based decoder, used for generating captions. Taking inspiration from the analysis of neural networks' robustness against adversarial perturbations, we propose a novel gray-box algorithm for creating adversarial examples in image-to-text models. Unlike image classification tasks that have a finite set of class labels, finding visually similar adversarial examples in an image-to-text task poses greater challenges because the captioning system allows for a virtually infinite space of possible captions. In this paper, we present a gray-box adversarial attack on image-to-text, both untargeted and targeted. We formulate the process of discovering adversarial perturbations as an optimization problem that uses only the image-encoder component, meaning the proposed attack is language-model agnostic. Through experiments conducted on the ViT-GPT2 model, which is the most-used image-to-text model in Hugging Face, and the Flickr30k dataset, we demonstrate that our proposed attack successfully generates visually similar adversarial examples, both with untargeted and targeted captions. Notably, our attack operates in a gray-box manner, requiring no knowledge about the decoder module. We also show that our attacks fool the popular open-source platform Hugging Face.



## **16. Random-Set Convolutional Neural Network (RS-CNN) for Epistemic Deep Learning**

cs.LG

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2307.05772v1) [paper-pdf](http://arxiv.org/pdf/2307.05772v1)

**Authors**: Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, Fabio Cuzzolin

**Abstract**: Machine learning is increasingly deployed in safety-critical domains where robustness against adversarial attacks is crucial and erroneous predictions could lead to potentially catastrophic consequences. This highlights the need for learning systems to be equipped with the means to determine a model's confidence in its prediction and the epistemic uncertainty associated with it, 'to know when a model does not know'. In this paper, we propose a novel Random-Set Convolutional Neural Network (RS-CNN) for classification which predicts belief functions rather than probability vectors over the set of classes, using the mathematics of random sets, i.e., distributions over the power set of the sample space. Based on the epistemic deep learning approach, random-set models are capable of representing the 'epistemic' uncertainty induced in machine learning by limited training sets. We estimate epistemic uncertainty by approximating the size of credal sets associated with the predicted belief functions, and experimentally demonstrate how our approach outperforms competing uncertainty-aware approaches in a classical evaluation setting. The performance of RS-CNN is best demonstrated on OOD samples where it manages to capture the true prediction while standard CNNs fail.



## **17. In and Out-of-Domain Text Adversarial Robustness via Label Smoothing**

cs.CL

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2212.10258v2) [paper-pdf](http://arxiv.org/pdf/2212.10258v2)

**Authors**: Yahan Yang, Soham Dan, Dan Roth, Insup Lee

**Abstract**: Recently it has been shown that state-of-the-art NLP models are vulnerable to adversarial attacks, where the predictions of a model can be drastically altered by slight modifications to the input (such as synonym substitutions). While several defense techniques have been proposed, and adapted, to the discrete nature of text adversarial attacks, the benefits of general-purpose regularization methods such as label smoothing for language models, have not been studied. In this paper, we study the adversarial robustness provided by various label smoothing strategies in foundational models for diverse NLP tasks in both in-domain and out-of-domain settings. Our experiments show that label smoothing significantly improves adversarial robustness in pre-trained models like BERT, against various popular attacks. We also analyze the relationship between prediction confidence and robustness, showing that label smoothing reduces over-confident errors on adversarial examples.



## **18. Zeroth-order Optimization with Weak Dimension Dependency**

math.OC

to be published in COLT 2023

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2307.05753v1) [paper-pdf](http://arxiv.org/pdf/2307.05753v1)

**Authors**: Pengyun Yue, Long Yang, Cong Fang, Zhouchen Lin

**Abstract**: Zeroth-order optimization is a fundamental research topic that has been a focus of various learning tasks, such as black-box adversarial attacks, bandits, and reinforcement learning. However, in theory, most complexity results assert a linear dependency on the dimension of optimization variable, which implies paralyzations of zeroth-order algorithms for high-dimensional problems and cannot explain their effectiveness in practice. In this paper, we present a novel zeroth-order optimization theory characterized by complexities that exhibit weak dependencies on dimensionality. The key contribution lies in the introduction of a new factor, denoted as $\mathrm{ED}_{\alpha}=\sup_{x\in \mathbb{R}^d}\sum_{i=1}^d\sigma_i^\alpha(\nabla^2 f(x))$ ($\alpha>0$, $\sigma_i(\cdot)$ is the $i$-th singular value in non-increasing order), which effectively functions as a measure of dimensionality. The algorithms we propose demonstrate significantly reduced complexities when measured in terms of the factor $\mathrm{ED}_{\alpha}$. Specifically, we first study a well-known zeroth-order algorithm from Nesterov and Spokoiny (2017) on quadratic objectives and show a complexity of $\mathcal{O}\left(\frac{\mathrm{ED}_1}{\sigma_d}\log(1/\epsilon)\right)$ for the strongly convex setting. Furthermore, we introduce novel algorithms that leverages the Heavy-ball mechanism. Our proposed algorithm exhibits a complexity of $\mathcal{O}\left(\frac{\mathrm{ED}_{1/2}}{\sqrt{\sigma_d}}\cdot\log{\frac{L}{\mu}}\cdot\log(1/\epsilon)\right)$. We further expand the scope of the method to encompass generic smooth optimization problems under an additional Hessian-smooth condition. The resultant algorithms demonstrate remarkable complexities which improve by an order in $d$ under appropriate conditions. Our analysis lays the foundation for zeroth-order optimization methods for smooth functions within high-dimensional settings.



## **19. Adversarial Cheap Talk**

cs.LG

To be published at ICML 2023. Project video and code are available at  https://sites.google.com/view/adversarial-cheap-talk

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2211.11030v4) [paper-pdf](http://arxiv.org/pdf/2211.11030v4)

**Authors**: Chris Lu, Timon Willi, Alistair Letcher, Jakob Foerster

**Abstract**: Adversarial attacks in reinforcement learning (RL) often assume highly-privileged access to the victim's parameters, environment, or data. Instead, this paper proposes a novel adversarial setting called a Cheap Talk MDP in which an Adversary can merely append deterministic messages to the Victim's observation, resulting in a minimal range of influence. The Adversary cannot occlude ground truth, influence underlying environment dynamics or reward signals, introduce non-stationarity, add stochasticity, see the Victim's actions, or access their parameters. Additionally, we present a simple meta-learning algorithm called Adversarial Cheap Talk (ACT) to train Adversaries in this setting. We demonstrate that an Adversary trained with ACT still significantly influences the Victim's training and testing performance, despite the highly constrained setting. Affecting train-time performance reveals a new attack vector and provides insight into the success and failure modes of existing RL algorithms. More specifically, we show that an ACT Adversary is capable of harming performance by interfering with the learner's function approximation, or instead helping the Victim's performance by outputting useful features. Finally, we show that an ACT Adversary can manipulate messages during train-time to directly and arbitrarily control the Victim at test-time. Project video and code are available at https://sites.google.com/view/adversarial-cheap-talk



## **20. Hyper-parameter Tuning for Adversarially Robust Models**

cs.LG

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2304.02497v2) [paper-pdf](http://arxiv.org/pdf/2304.02497v2)

**Authors**: Pedro Mendes, Paolo Romano, David Garlan

**Abstract**: This work focuses on the problem of hyper-parameter tuning (HPT) for robust (i.e., adversarially trained) models, shedding light on the new challenges and opportunities arising during the HPT process for robust models. To this end, we conduct an extensive experimental study based on 3 popular deep models, in which we explore exhaustively 9 (discretized) HPs, 2 fidelity dimensions, and 2 attack bounds, for a total of 19208 configurations (corresponding to 50 thousand GPU hours). Through this study, we show that the complexity of the HPT problem is further exacerbated in adversarial settings due to the need to independently tune the HPs used during standard and adversarial training: succeeding in doing so (i.e., adopting different HP settings in both phases) can lead to a reduction of up to 80% and 43% of the error for clean and adversarial inputs, respectively. On the other hand, we also identify new opportunities to reduce the cost of HPT for robust models. Specifically, we propose to leverage cheap adversarial training methods to obtain inexpensive, yet highly correlated, estimations of the quality achievable using state-of-the-art methods. We show that, by exploiting this novel idea in conjunction with a recent multi-fidelity optimizer (taKG), the efficiency of the HPT process can be enhanced by up to 2.1x.



## **21. Revisiting the Trade-off between Accuracy and Robustness via Weight Distribution of Filters**

cs.CV

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2306.03430v2) [paper-pdf](http://arxiv.org/pdf/2306.03430v2)

**Authors**: Xingxing Wei, Shiji Zhao

**Abstract**: Adversarial attacks have been proven to be potential threats to Deep Neural Networks (DNNs), and many methods are proposed to defend against adversarial attacks. However, while enhancing the robustness, the clean accuracy will decline to a certain extent, implying a trade-off existed between the accuracy and robustness. In this paper, we firstly empirically find an obvious distinction between standard and robust models in the filters' weight distribution of the same architecture, and then theoretically explain this phenomenon in terms of the gradient regularization, which shows this difference is an intrinsic property for DNNs, and thus a static network architecture is difficult to improve the accuracy and robustness at the same time. Secondly, based on this observation, we propose a sample-wise dynamic network architecture named Adversarial Weight-Varied Network (AW-Net), which focuses on dealing with clean and adversarial examples with a ``divide and rule" weight strategy. The AW-Net dynamically adjusts network's weights based on regulation signals generated by an adversarial detector, which is directly influenced by the input sample. Benefiting from the dynamic network architecture, clean and adversarial examples can be processed with different network weights, which provides the potentiality to enhance the accuracy and robustness simultaneously. A series of experiments demonstrate that our AW-Net is architecture-friendly to handle both clean and adversarial examples and can achieve better trade-off performance than state-of-the-art robust models.



## **22. Mitigating the Accuracy-Robustness Trade-off via Multi-Teacher Adversarial Distillation**

cs.LG

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2306.16170v2) [paper-pdf](http://arxiv.org/pdf/2306.16170v2)

**Authors**: Shiji Zhao, Xizhe Wang, Xingxing Wei

**Abstract**: Adversarial training is a practical approach for improving the robustness of deep neural networks against adversarial attacks. Although bringing reliable robustness, the performance toward clean examples is negatively affected after adversarial training, which means a trade-off exists between accuracy and robustness. Recently, some studies have tried to use knowledge distillation methods in adversarial training, achieving competitive performance in improving the robustness but the accuracy for clean samples is still limited. In this paper, to mitigate the accuracy-robustness trade-off, we introduce the Multi-Teacher Adversarial Robustness Distillation (MTARD) to guide the model's adversarial training process by applying a strong clean teacher and a strong robust teacher to handle the clean examples and adversarial examples, respectively. During the optimization process, to ensure that different teachers show similar knowledge scales, we design the Entropy-Based Balance algorithm to adjust the teacher's temperature and keep the teachers' information entropy consistent. Besides, to ensure that the student has a relatively consistent learning speed from multiple teachers, we propose the Normalization Loss Balance algorithm to adjust the learning weights of different types of knowledge. A series of experiments conducted on public datasets demonstrate that MTARD outperforms the state-of-the-art adversarial training and distillation methods against various adversarial attacks.



## **23. Membership Inference Attacks on DNNs using Adversarial Perturbations**

cs.LG

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2307.05193v1) [paper-pdf](http://arxiv.org/pdf/2307.05193v1)

**Authors**: Hassan Ali, Adnan Qayyum, Ala Al-Fuqaha, Junaid Qadir

**Abstract**: Several membership inference (MI) attacks have been proposed to audit a target DNN. Given a set of subjects, MI attacks tell which subjects the target DNN has seen during training. This work focuses on the post-training MI attacks emphasizing high confidence membership detection -- True Positive Rates (TPR) at low False Positive Rates (FPR). Current works in this category -- likelihood ratio attack (LiRA) and enhanced MI attack (EMIA) -- only perform well on complex datasets (e.g., CIFAR-10 and Imagenet) where the target DNN overfits its train set, but perform poorly on simpler datasets (0% TPR by both attacks on Fashion-MNIST, 2% and 0% TPR respectively by LiRA and EMIA on MNIST at 1% FPR). To address this, firstly, we unify current MI attacks by presenting a framework divided into three stages -- preparation, indication and decision. Secondly, we utilize the framework to propose two novel attacks: (1) Adversarial Membership Inference Attack (AMIA) efficiently utilizes the membership and the non-membership information of the subjects while adversarially minimizing a novel loss function, achieving 6% TPR on both Fashion-MNIST and MNIST datasets; and (2) Enhanced AMIA (E-AMIA) combines EMIA and AMIA to achieve 8% and 4% TPRs on Fashion-MNIST and MNIST datasets respectively, at 1% FPR. Thirdly, we introduce two novel augmented indicators that positively leverage the loss information in the Gaussian neighborhood of a subject. This improves TPR of all four attacks on average by 2.5% and 0.25% respectively on Fashion-MNIST and MNIST datasets at 1% FPR. Finally, we propose simple, yet novel, evaluation metric, the running TPR average (RTA) at a given FPR, that better distinguishes different MI attacks in the low FPR region. We also show that AMIA and E-AMIA are more transferable to the unknown DNNs (other than the target DNN) and are more robust to DP-SGD training as compared to LiRA and EMIA.



## **24. ATWM: Defense against adversarial malware based on adversarial training**

cs.CR

**SubmitDate**: 2023-07-11    [abs](http://arxiv.org/abs/2307.05095v1) [paper-pdf](http://arxiv.org/pdf/2307.05095v1)

**Authors**: Kun Li, Fan Zhang, Wei Guo

**Abstract**: Deep learning technology has made great achievements in the field of image. In order to defend against malware attacks, researchers have proposed many Windows malware detection models based on deep learning. However, deep learning models are vulnerable to adversarial example attacks. Malware can generate adversarial malware with the same malicious function to attack the malware detection model and evade detection of the model. Currently, many adversarial defense studies have been proposed, but existing adversarial defense studies are based on image sample and cannot be directly applied to malware sample. Therefore, this paper proposes an adversarial malware defense method based on adversarial training. This method uses preprocessing to defend simple adversarial examples to reduce the difficulty of adversarial training. Moreover, this method improves the adversarial defense capability of the model through adversarial training. We experimented with three attack methods in two sets of datasets, and the results show that the method in this paper can improve the adversarial defense capability of the model without reducing the accuracy of the model.



## **25. Categorical composable cryptography: extended version**

cs.CR

Extended version of arXiv:2105.05949 which appeared in FoSSaCS 2022

**SubmitDate**: 2023-07-10    [abs](http://arxiv.org/abs/2208.13232v2) [paper-pdf](http://arxiv.org/pdf/2208.13232v2)

**Authors**: Anne Broadbent, Martti Karvonen

**Abstract**: We formalize the simulation paradigm of cryptography in terms of category theory and show that protocols secure against abstract attacks form a symmetric monoidal category, thus giving an abstract model of composable security definitions in cryptography. Our model is able to incorporate computational security, set-up assumptions and various attack models such as colluding or independently acting subsets of adversaries in a modular, flexible fashion. We conclude by using string diagrams to rederive the security of the one-time pad, correctness of Diffie-Hellman key exchange and no-go results concerning the limits of bipartite and tripartite cryptography, ruling out e.g., composable commitments and broadcasting. On the way, we exhibit two categorical constructions of resource theories that might be of independent interest: one capturing resources shared among multiple parties and one capturing resource conversions that succeed asymptotically.



## **26. On the Robustness of Bayesian Neural Networks to Adversarial Attacks**

cs.LG

arXiv admin note: text overlap with arXiv:2002.04359

**SubmitDate**: 2023-07-10    [abs](http://arxiv.org/abs/2207.06154v2) [paper-pdf](http://arxiv.org/pdf/2207.06154v2)

**Authors**: Luca Bortolussi, Ginevra Carbone, Luca Laurenti, Andrea Patane, Guido Sanguinetti, Matthew Wicker

**Abstract**: Vulnerability to adversarial attacks is one of the principal hurdles to the adoption of deep learning in safety-critical applications. Despite significant efforts, both practical and theoretical, training deep learning models robust to adversarial attacks is still an open problem. In this paper, we analyse the geometry of adversarial attacks in the large-data, overparameterized limit for Bayesian Neural Networks (BNNs). We show that, in the limit, vulnerability to gradient-based attacks arises as a result of degeneracy in the data distribution, i.e., when the data lies on a lower-dimensional submanifold of the ambient space. As a direct consequence, we demonstrate that in this limit BNN posteriors are robust to gradient-based adversarial attacks. Crucially, we prove that the expected gradient of the loss with respect to the BNN posterior distribution is vanishing, even when each neural network sampled from the posterior is vulnerable to gradient-based attacks. Experimental results on the MNIST, Fashion MNIST, and half moons datasets, representing the finite data regime, with BNNs trained with Hamiltonian Monte Carlo and Variational Inference, support this line of arguments, showing that BNNs can display both high accuracy on clean data and robustness to both gradient-based and gradient-free based adversarial attacks.



## **27. Enhancing Adversarial Robustness via Score-Based Optimization**

cs.LG

**SubmitDate**: 2023-07-10    [abs](http://arxiv.org/abs/2307.04333v1) [paper-pdf](http://arxiv.org/pdf/2307.04333v1)

**Authors**: Boya Zhang, Weijian Luo, Zhihua Zhang

**Abstract**: Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.



## **28. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

cs.CV

24 pages, 17 figures, 1 table

**SubmitDate**: 2023-07-10    [abs](http://arxiv.org/abs/2307.02881v2) [paper-pdf](http://arxiv.org/pdf/2307.02881v2)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating probability density functions for images that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space - not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, although images may lie on such lower-dimensional manifolds, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. In pursuing this goal, we consider generative models that are popular in AI and computer vision community. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: it should be possible to sample from this distribution according to the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute the probability of the sample, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show that such probabilistic descriptions can be used to construct defences against adversarial attacks. In addition to describing the manifold in terms of density, we also consider how semantic interpretations can be used to describe points on the manifold. To this end, we consider an emergent language framework which makes use of variational encoders to produce a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described in terms of evolving semantic descriptions.



## **29. Testing Robustness Against Unforeseen Adversaries**

cs.LG

Datasets available at  https://github.com/centerforaisafety/adversarial-corruptions

**SubmitDate**: 2023-07-09    [abs](http://arxiv.org/abs/1908.08016v3) [paper-pdf](http://arxiv.org/pdf/1908.08016v3)

**Authors**: Max Kaufmann, Daniel Kang, Yi Sun, Steven Basart, Xuwang Yin, Mantas Mazeika, Akul Arora, Adam Dziedzic, Franziska Boenisch, Tom Brown, Jacob Steinhardt, Dan Hendrycks

**Abstract**: When considering real-world adversarial settings, defenders are unlikely to have access to the full range of deployment-time adversaries during training, and adversaries are likely to use realistic adversarial distortions that will not be limited to small L_p-constrained perturbations. To narrow in on this discrepancy between research and reality we introduce eighteen novel adversarial attacks, which we use to create ImageNet-UA, a new benchmark for evaluating model robustness against a wide range of unforeseen adversaries. We make use of our benchmark to identify a range of defense strategies which can help overcome this generalization gap, finding a rich space of techniques which can improve unforeseen robustness. We hope the greater variety and realism of ImageNet-UA will make it a useful tool for those working on real-world worst-case robustness, enabling development of more robust defenses which can generalize beyond attacks seen during training.



## **30. GNP Attack: Transferable Adversarial Examples via Gradient Norm Penalty**

cs.LG

30th IEEE International Conference on Image Processing (ICIP),  October 2023

**SubmitDate**: 2023-07-09    [abs](http://arxiv.org/abs/2307.04099v1) [paper-pdf](http://arxiv.org/pdf/2307.04099v1)

**Authors**: Tao Wu, Tie Luo, Donald C. Wunsch

**Abstract**: Adversarial examples (AE) with good transferability enable practical black-box attacks on diverse target models, where insider knowledge about the target models is not required. Previous methods often generate AE with no or very limited transferability; that is, they easily overfit to the particular architecture and feature representation of the source, white-box model and the generated AE barely work for target, black-box models. In this paper, we propose a novel approach to enhance AE transferability using Gradient Norm Penalty (GNP). It drives the loss function optimization procedure to converge to a flat region of local optima in the loss landscape. By attacking 11 state-of-the-art (SOTA) deep learning models and 6 advanced defense methods, we empirically show that GNP is very effective in generating AE with high transferability. We also demonstrate that it is very flexible in that it can be easily integrated with other gradient based methods for stronger transfer-based attacks.



## **31. Random Position Adversarial Patch for Vision Transformers**

cs.CV

**SubmitDate**: 2023-07-09    [abs](http://arxiv.org/abs/2307.04066v1) [paper-pdf](http://arxiv.org/pdf/2307.04066v1)

**Authors**: Mingzhen Shao

**Abstract**: Previous studies have shown the vulnerability of vision transformers to adversarial patches, but these studies all rely on a critical assumption: the attack patches must be perfectly aligned with the patches used for linear projection in vision transformers. Due to this stringent requirement, deploying adversarial patches for vision transformers in the physical world becomes impractical, unlike their effectiveness on CNNs. This paper proposes a novel method for generating an adversarial patch (G-Patch) that overcomes the alignment constraint, allowing the patch to launch a targeted attack at any position within the field of view. Specifically, instead of directly optimizing the patch using gradients, we employ a GAN-like structure to generate the adversarial patch. Our experiments show the effectiveness of the adversarial patch in achieving universal attacks on vision transformers, both in digital and physical-world scenarios. Additionally, further analysis reveals that the generated adversarial patch exhibits robustness to brightness restriction, color transfer, and random noise. Real-world attack experiments validate the effectiveness of the G-Patch to launch robust attacks even under some very challenging conditions.



## **32. Robust Ranking Explanations**

cs.LG

Accepted to IMLH (Interpretable ML in Healthcare) workshop at ICML  2023. arXiv admin note: substantial text overlap with arXiv:2212.14106

**SubmitDate**: 2023-07-08    [abs](http://arxiv.org/abs/2307.04024v1) [paper-pdf](http://arxiv.org/pdf/2307.04024v1)

**Authors**: Chao Chen, Chenghua Guo, Guixiang Ma, Ming Zeng, Xi Zhang, Sihong Xie

**Abstract**: Robust explanations of machine learning models are critical to establish human trust in the models. Due to limited cognition capability, most humans can only interpret the top few salient features. It is critical to make top salient features robust to adversarial attacks, especially those against the more vulnerable gradient-based explanations. Existing defense measures robustness using $\ell_p$-norms, which have weaker protection power. We define explanation thickness for measuring salient features ranking stability, and derive tractable surrogate bounds of the thickness to design the \textit{R2ET} algorithm to efficiently maximize the thickness and anchor top salient features. Theoretically, we prove a connection between R2ET and adversarial training. Experiments with a wide spectrum of network architectures and data modalities, including brain networks, demonstrate that R2ET attains higher explanation robustness under stealthy attacks while retaining accuracy.



## **33. Provable Robust Saliency-based Explanations**

cs.LG

**SubmitDate**: 2023-07-08    [abs](http://arxiv.org/abs/2212.14106v3) [paper-pdf](http://arxiv.org/pdf/2212.14106v3)

**Authors**: Chao Chen, Chenghua Guo, Guixiang Ma, Ming Zeng, Xi Zhang, Sihong Xie

**Abstract**: Robust explanations of machine learning models are critical to establishing human trust in the models. The top-$k$ intersection is widely used to evaluate the robustness of explanations. However, most existing attacking and defense strategies are based on $\ell_p$ norms, thus creating a mismatch between the evaluation and optimization objectives. To this end, we define explanation thickness for measuring top-$k$ salient features ranking stability, and design the \textit{R2ET} algorithm based on a novel tractable surrogate to maximize the thickness and stabilize the top salient features efficiently. Theoretically, we prove a connection between R2ET and adversarial training; using a novel multi-objective optimization formulation and a generalization error bound, we further prove that the surrogate objective can improve both the numerical and statistical stability of the explanations. Experiments with a wide spectrum of network architectures and data modalities demonstrate that R2ET attains higher explanation robustness under stealthy attacks while retaining model accuracy.



## **34. Adversarial Self-Attack Defense and Spatial-Temporal Relation Mining for Visible-Infrared Video Person Re-Identification**

cs.CV

11 pages,8 figures

**SubmitDate**: 2023-07-08    [abs](http://arxiv.org/abs/2307.03903v1) [paper-pdf](http://arxiv.org/pdf/2307.03903v1)

**Authors**: Huafeng Li, Le Xu, Yafei Zhang, Dapeng Tao, Zhengtao Yu

**Abstract**: In visible-infrared video person re-identification (re-ID), extracting features not affected by complex scenes (such as modality, camera views, pedestrian pose, background, etc.) changes, and mining and utilizing motion information are the keys to solving cross-modal pedestrian identity matching. To this end, the paper proposes a new visible-infrared video person re-ID method from a novel perspective, i.e., adversarial self-attack defense and spatial-temporal relation mining. In this work, the changes of views, posture, background and modal discrepancy are considered as the main factors that cause the perturbations of person identity features. Such interference information contained in the training samples is used as an adversarial perturbation. It performs adversarial attacks on the re-ID model during the training to make the model more robust to these unfavorable factors. The attack from the adversarial perturbation is introduced by activating the interference information contained in the input samples without generating adversarial samples, and it can be thus called adversarial self-attack. This design allows adversarial attack and defense to be integrated into one framework. This paper further proposes a spatial-temporal information-guided feature representation network to use the information in video sequences. The network cannot only extract the information contained in the video-frame sequences but also use the relation of the local information in space to guide the network to extract more robust features. The proposed method exhibits compelling performance on large-scale cross-modality video datasets. The source code of the proposed method will be released at https://github.com/lhf12278/xxx.



## **35. On Pseudolinear Codes for Correcting Adversarial Errors**

cs.IT

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2307.05528v1) [paper-pdf](http://arxiv.org/pdf/2307.05528v1)

**Authors**: Eric Ruzomberka, Homa Nikbakht, Christopher G. Brinton, H. Vincent Poor

**Abstract**: We consider error-correction coding schemes for adversarial wiretap channels (AWTCs) in which the channel can a) read a fraction of the codeword bits up to a bound $r$ and b) flip a fraction of the bits up to a bound $p$. The channel can freely choose the locations of the bit reads and bit flips via a process with unbounded computational power. Codes for the AWTC are of broad interest in the area of information security, as they can provide data resiliency in settings where an attacker has limited access to a storage or transmission medium.   We investigate a family of non-linear codes known as pseudolinear codes, which were first proposed by Guruswami and Indyk (FOCS 2001) for constructing list-decodable codes independent of the AWTC setting. Unlike general non-linear codes, pseudolinear codes admit efficient encoders and have succinct representations. We focus on unique decoding and show that random pseudolinear codes can achieve rates up to the binary symmetric channel (BSC) capacity $1-H_2(p)$ for any $p,r$ in the less noisy region: $p<1/2$ and $r<1-H_2(p)$ where $H_2(\cdot)$ is the binary entropy function. Thus, pseudolinear codes are the first known optimal-rate binary code family for the less noisy AWTC that admit efficient encoders. The above result can be viewed as a derandomization result of random general codes in the AWTC setting, which in turn opens new avenues for applying derandomization techniques to randomized constructions of AWTC codes. Our proof applies a novel concentration inequality for sums of random variables with limited independence which may be of interest as an analysis tool more generally.



## **36. A Theoretical Perspective on Subnetwork Contributions to Adversarial Robustness**

cs.LG

3 figures, 3 tables, 17 pages, has appendices

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2307.03803v1) [paper-pdf](http://arxiv.org/pdf/2307.03803v1)

**Authors**: Jovon Craig, Josh Andle, Theodore S. Nowak, Salimeh Yasaei Sekeh

**Abstract**: The robustness of deep neural networks (DNNs) against adversarial attacks has been studied extensively in hopes of both better understanding how deep learning models converge and in order to ensure the security of these models in safety-critical applications. Adversarial training is one approach to strengthening DNNs against adversarial attacks, and has been shown to offer a means for doing so at the cost of applying computationally expensive training methods to the entire model. To better understand these attacks and facilitate more efficient adversarial training, in this paper we develop a novel theoretical framework that investigates how the adversarial robustness of a subnetwork contributes to the robustness of the entire network. To do so we first introduce the concept of semirobustness, which is a measure of the adversarial robustness of a subnetwork. Building on this concept, we then provide a theoretical analysis to show that if a subnetwork is semirobust and there is a sufficient dependency between it and each subsequent layer in the network, then the remaining layers are also guaranteed to be robust. We validate these findings empirically across multiple DNN architectures, datasets, and adversarial attacks. Experiments show the ability of a robust subnetwork to promote full-network robustness, and investigate the layer-wise dependencies required for this full-network robustness to be achieved.



## **37. When and How to Fool Explainable Models (and Humans) with Adversarial Examples**

cs.LG

Updated version. 43 pages, 9 figures, 4 tables

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2107.01943v2) [paper-pdf](http://arxiv.org/pdf/2107.01943v2)

**Authors**: Jon Vadillo, Roberto Santana, Jose A. Lozano

**Abstract**: Reliable deployment of machine learning models such as neural networks continues to be challenging due to several limitations. Some of the main shortcomings are the lack of interpretability and the lack of robustness against adversarial examples or out-of-distribution inputs. In this exploratory review, we explore the possibilities and limits of adversarial attacks for explainable machine learning models. First, we extend the notion of adversarial examples to fit in explainable machine learning scenarios, in which the inputs, the output classifications and the explanations of the model's decisions are assessed by humans. Next, we propose a comprehensive framework to study whether (and how) adversarial examples can be generated for explainable models under human assessment, introducing and illustrating novel attack paradigms. In particular, our framework considers a wide range of relevant yet often ignored factors such as the type of problem, the user expertise or the objective of the explanations, in order to identify the attack strategies that should be adopted in each scenario to successfully deceive the model (and the human). The intention of these contributions is to serve as a basis for a more rigorous and realistic study of adversarial examples in the field of explainable machine learning.



## **38. Enhancing Adversarial Training via Reweighting Optimization Trajectory**

cs.LG

Accepted by ECML 2023

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2306.14275v3) [paper-pdf](http://arxiv.org/pdf/2306.14275v3)

**Authors**: Tianjin Huang, Shiwei Liu, Tianlong Chen, Meng Fang, Li Shen, Vlaod Menkovski, Lu Yin, Yulong Pei, Mykola Pechenizkiy

**Abstract**: Despite the fact that adversarial training has become the de facto method for improving the robustness of deep neural networks, it is well-known that vanilla adversarial training suffers from daunting robust overfitting, resulting in unsatisfactory robust generalization. A number of approaches have been proposed to address these drawbacks such as extra regularization, adversarial weights perturbation, and training with more data over the last few years. However, the robust generalization improvement is yet far from satisfactory. In this paper, we approach this challenge with a brand new perspective -- refining historical optimization trajectories. We propose a new method named \textbf{Weighted Optimization Trajectories (WOT)} that leverages the optimization trajectories of adversarial training in time. We have conducted extensive experiments to demonstrate the effectiveness of WOT under various state-of-the-art adversarial attacks. Our results show that WOT integrates seamlessly with the existing adversarial training methods and consistently overcomes the robust overfitting issue, resulting in better adversarial robustness. For example, WOT boosts the robust accuracy of AT-PGD under AA-$L_{\infty}$ attack by 1.53\% $\sim$ 6.11\% and meanwhile increases the clean accuracy by 0.55\%$\sim$5.47\% across SVHN, CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.



## **39. Evaluating Similitude and Robustness of Deep Image Denoising Models via Adversarial Attack**

cs.CV

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2306.16050v2) [paper-pdf](http://arxiv.org/pdf/2306.16050v2)

**Authors**: Jie Ning, Jiebao Sun, Yao Li, Zhichang Guo, Wangmeng Zuo

**Abstract**: Deep neural networks (DNNs) have shown superior performance comparing to traditional image denoising algorithms. However, DNNs are inevitably vulnerable while facing adversarial attacks. In this paper, we propose an adversarial attack method named denoising-PGD which can successfully attack all the current deep denoising models while keep the noise distribution almost unchanged. We surprisingly find that the current mainstream non-blind denoising models (DnCNN, FFDNet, ECNDNet, BRDNet), blind denoising models (DnCNN-B, Noise2Noise, RDDCNN-B, FAN), plug-and-play (DPIR, CurvPnP) and unfolding denoising models (DeamNet) almost share the same adversarial sample set on both grayscale and color images, respectively. Shared adversarial sample set indicates that all these models are similar in term of local behaviors at the neighborhood of all the test samples. Thus, we further propose an indicator to measure the local similarity of models, called robustness similitude. Non-blind denoising models are found to have high robustness similitude across each other, while hybrid-driven models are also found to have high robustness similitude with pure data-driven non-blind denoising models. According to our robustness assessment, data-driven non-blind denoising models are the most robust. We use adversarial training to complement the vulnerability to adversarial attacks. Moreover, the model-driven image denoising BM3D shows resistance on adversarial attacks.



## **40. A Vulnerability of Attribution Methods Using Pre-Softmax Scores**

cs.LG

7 pages, 5 figures,

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03305v1) [paper-pdf](http://arxiv.org/pdf/2307.03305v1)

**Authors**: Miguel Lerma, Mirtha Lucas

**Abstract**: We discuss a vulnerability involving a category of attribution methods used to provide explanations for the outputs of convolutional neural networks working as classifiers. It is known that this type of networks are vulnerable to adversarial attacks, in which imperceptible perturbations of the input may alter the outputs of the model. In contrast, here we focus on effects that small modifications in the model may cause on the attribution method without altering the model outputs.



## **41. Quantum Solutions to the Privacy vs. Utility Tradeoff**

quant-ph

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03118v1) [paper-pdf](http://arxiv.org/pdf/2307.03118v1)

**Authors**: Sagnik Chatterjee, Vyacheslav Kungurtsev

**Abstract**: In this work, we propose a novel architecture (and several variants thereof) based on quantum cryptographic primitives with provable privacy and security guarantees regarding membership inference attacks on generative models. Our architecture can be used on top of any existing classical or quantum generative models. We argue that the use of quantum gates associated with unitary operators provides inherent advantages compared to standard Differential Privacy based techniques for establishing guaranteed security from all polynomial-time adversaries.



## **42. On Distribution-Preserving Mitigation Strategies for Communication under Cognitive Adversaries**

cs.IT

Presented at IEEE ISIT 2023

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03105v1) [paper-pdf](http://arxiv.org/pdf/2307.03105v1)

**Authors**: Soumita Hazra, J. Harshan

**Abstract**: In wireless security, cognitive adversaries are known to inject jamming energy on the victim's frequency band and monitor the same band for countermeasures thereby trapping the victim. Under the class of cognitive adversaries, we propose a new threat model wherein the adversary, upon executing the jamming attack, measures the long-term statistic of Kullback-Leibler Divergence (KLD) between its observations over each of the network frequencies before and after the jamming attack. To mitigate this adversary, we propose a new cooperative strategy wherein the victim takes the assistance for a helper node in the network to reliably communicate its message to the destination. The underlying idea is to appropriately split their energy and time resources such that their messages are reliably communicated without disturbing the statistical distribution of the samples in the network. We present rigorous analyses on the reliability and the covertness metrics at the destination and the adversary, respectively, and then synthesize tractable algorithms to obtain near-optimal division of resources between the victim and the helper. Finally, we show that the obtained near-optimal division of energy facilitates in deceiving the adversary with a KLD estimator.



## **43. NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic**

cs.CL

Published as a conference paper at ACL 2023

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02849v1) [paper-pdf](http://arxiv.org/pdf/2307.02849v1)

**Authors**: Zi'ou Zheng, Xiaodan Zhu

**Abstract**: Reasoning has been a central topic in artificial intelligence from the beginning. The recent progress made on distributed representation and neural networks continues to improve the state-of-the-art performance of natural language inference. However, it remains an open question whether the models perform real reasoning to reach their conclusions or rely on spurious correlations. Adversarial attacks have proven to be an important tool to help evaluate the Achilles' heel of the victim models. In this study, we explore the fundamental problem of developing attack models based on logic formalism. We propose NatLogAttack to perform systematic attacks centring around natural logic, a classical logic formalism that is traceable back to Aristotle's syllogism and has been closely developed for natural language inference. The proposed framework renders both label-preserving and label-flipping attacks. We show that compared to the existing attack models, NatLogAttack generates better adversarial examples with fewer visits to the victim models. The victim models are found to be more vulnerable under the label-flipping setting. NatLogAttack provides a tool to probe the existing and future NLI models' capacity from a key viewpoint and we hope more logic-based attacks will be further explored for understanding the desired property of reasoning.



## **44. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

cs.CV

10 pages, 6 figures, 7 tables. arXiv admin note: substantial text  overlap with arXiv:2204.02887

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02828v1) [paper-pdf](http://arxiv.org/pdf/2307.02828v1)

**Authors**: Xu Han, Anmin Liu, Chenxuan Yao, Yanbo Fan, Kun He

**Abstract**: Deep neural networks are known to be vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to the benign input. After achieving nearly 100% attack success rates in white-box setting, more focus is shifted to black-box attacks, of which the transferability of adversarial examples has gained significant attention. In either case, the common gradient-based methods generally use the sign function to generate perturbations on the gradient update, that offers a roughly correct direction and has gained great success. But little work pays attention to its possible limitation. In this work, we observe that the deviation between the original gradient and the generated noise may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability. To this end, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM). Specifically, we use data rescaling to substitute the sign function without extra computational cost. We further propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method could be used in any gradient-based attacks and is extensible to be integrated with various input transformation or ensemble methods to further improve the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our method could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.



## **45. A Testbed To Study Adversarial Cyber-Attack Strategies in Enterprise Networks**

cs.CR

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02794v1) [paper-pdf](http://arxiv.org/pdf/2307.02794v1)

**Authors**: Ayush Kumar, David K. Yau

**Abstract**: In this work, we propose a testbed environment to capture the attack strategies of an adversary carrying out a cyber-attack on an enterprise network. The testbed contains nodes with known security vulnerabilities which can be exploited by hackers. Participants can be invited to play the role of a hacker (e.g., black-hat, hacktivist) and attack the testbed. The testbed is designed such that there are multiple attack pathways available to hackers. We describe the working of the testbed components and discuss its implementation on a VMware ESXi server. Finally, we subject our testbed implementation to a few well-known cyber-attack strategies, collect data during the process and present our analysis of the data.



## **46. Chaos Theory and Adversarial Robustness**

cs.LG

14 pages, 6 figures

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2210.13235v2) [paper-pdf](http://arxiv.org/pdf/2210.13235v2)

**Authors**: Jonathan S. Kent

**Abstract**: Neural networks, being susceptible to adversarial attacks, should face a strict level of scrutiny before being deployed in critical or adversarial applications. This paper uses ideas from Chaos Theory to explain, analyze, and quantify the degree to which neural networks are susceptible to or robust against adversarial attacks. To this end, we present a new metric, the "susceptibility ratio," given by $\hat \Psi(h, \theta)$, which captures how greatly a model's output will be changed by perturbations to a given input.   Our results show that susceptibility to attack grows significantly with the depth of the model, which has safety implications for the design of neural networks for production environments. We provide experimental evidence of the relationship between $\hat \Psi$ and the post-attack accuracy of classification models, as well as a discussion of its application to tasks lacking hard decision boundaries. We also demonstrate how to quickly and easily approximate the certified robustness radii for extremely large models, which until now has been computationally infeasible to calculate directly.



## **47. GIT: Detecting Uncertainty, Out-Of-Distribution and Adversarial Samples using Gradients and Invariance Transformations**

cs.LG

Accepted at IJCNN 2023

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02672v1) [paper-pdf](http://arxiv.org/pdf/2307.02672v1)

**Authors**: Julia Lust, Alexandru P. Condurache

**Abstract**: Deep neural networks tend to make overconfident predictions and often require additional detectors for misclassifications, particularly for safety-critical applications. Existing detection methods usually only focus on adversarial attacks or out-of-distribution samples as reasons for false predictions. However, generalization errors occur due to diverse reasons often related to poorly learning relevant invariances. We therefore propose GIT, a holistic approach for the detection of generalization errors that combines the usage of gradient information and invariance transformations. The invariance transformations are designed to shift misclassified samples back into the generalization area of the neural network, while the gradient information measures the contradiction between the initial prediction and the corresponding inherent computations of the neural network using the transformed sample. Our experiments demonstrate the superior performance of GIT compared to the state-of-the-art on a variety of network architectures, problem setups and perturbation types.



## **48. Jailbroken: How Does LLM Safety Training Fail?**

cs.LG

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02483v1) [paper-pdf](http://arxiv.org/pdf/2307.02483v1)

**Authors**: Alexander Wei, Nika Haghtalab, Jacob Steinhardt

**Abstract**: Large language models trained for safety and harmlessness remain susceptible to adversarial misuse, as evidenced by the prevalence of "jailbreak" attacks on early releases of ChatGPT that elicit undesired behavior. Going beyond recognition of the issue, we investigate why such attacks succeed and how they can be created. We hypothesize two failure modes of safety training: competing objectives and mismatched generalization. Competing objectives arise when a model's capabilities and safety goals conflict, while mismatched generalization occurs when safety training fails to generalize to a domain for which capabilities exist. We use these failure modes to guide jailbreak design and then evaluate state-of-the-art models, including OpenAI's GPT-4 and Anthropic's Claude v1.3, against both existing and newly designed attacks. We find that vulnerabilities persist despite the extensive red-teaming and safety-training efforts behind these models. Notably, new attacks utilizing our failure modes succeed on every prompt in a collection of unsafe requests from the models' red-teaming evaluation sets and outperform existing ad hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity -- that safety mechanisms should be as sophisticated as the underlying model -- and argues against the idea that scaling alone can resolve these safety failure modes.



## **49. Defense against Adversarial Cloud Attack on Remote Sensing Salient Object Detection**

cs.CV

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2306.17431v2) [paper-pdf](http://arxiv.org/pdf/2306.17431v2)

**Authors**: Huiming Sun, Lan Fu, Jinlong Li, Qing Guo, Zibo Meng, Tianyun Zhang, Yuewei Lin, Hongkai Yu

**Abstract**: Detecting the salient objects in a remote sensing image has wide applications for the interdisciplinary research. Many existing deep learning methods have been proposed for Salient Object Detection (SOD) in remote sensing images and get remarkable results. However, the recent adversarial attack examples, generated by changing a few pixel values on the original remote sensing image, could result in a collapse for the well-trained deep learning based SOD model. Different with existing methods adding perturbation to original images, we propose to jointly tune adversarial exposure and additive perturbation for attack and constrain image close to cloudy image as Adversarial Cloud. Cloud is natural and common in remote sensing images, however, camouflaging cloud based adversarial attack and defense for remote sensing images are not well studied before. Furthermore, we design DefenseNet as a learn-able pre-processing to the adversarial cloudy images so as to preserve the performance of the deep learning based remote sensing SOD model, without tuning the already deployed deep SOD model. By considering both regular and generalized adversarial examples, the proposed DefenseNet can defend the proposed Adversarial Cloud in white-box setting and other attack methods in black-box setting. Experimental results on a synthesized benchmark from the public remote sensing SOD dataset (EORSSD) show the promising defense against adversarial cloud attacks.



## **50. On the Adversarial Robustness of Generative Autoencoders in the Latent Space**

cs.LG

18 pages, 12 figures

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02202v1) [paper-pdf](http://arxiv.org/pdf/2307.02202v1)

**Authors**: Mingfei Lu, Badong Chen

**Abstract**: The generative autoencoders, such as the variational autoencoders or the adversarial autoencoders, have achieved great success in lots of real-world applications, including image generation, and signal communication.   However, little concern has been devoted to their robustness during practical deployment.   Due to the probabilistic latent structure, variational autoencoders (VAEs) may confront problems such as a mismatch between the posterior distribution of the latent and real data manifold, or discontinuity in the posterior distribution of the latent.   This leaves a back door for malicious attackers to collapse VAEs from the latent space, especially in scenarios where the encoder and decoder are used separately, such as communication and compressed sensing.   In this work, we provide the first study on the adversarial robustness of generative autoencoders in the latent space.   Specifically, we empirically demonstrate the latent vulnerability of popular generative autoencoders through attacks in the latent space.   We also evaluate the difference between variational autoencoders and their deterministic variants and observe that the latter performs better in latent robustness.   Meanwhile, we identify a potential trade-off between the adversarial robustness and the degree of the disentanglement of the latent codes.   Additionally, we also verify the feasibility of improvement for the latent robustness of VAEs through adversarial training.   In summary, we suggest concerning the adversarial latent robustness of the generative autoencoders, analyze several robustness-relative issues, and give some insights into a series of key challenges.



