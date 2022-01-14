# Latest Adversarial Attack Papers
**update at 2022-01-14 11:44:02**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Unlabeled Data Improves Adversarial Robustness**

未标记的数据提高了对手的健壮性 stat.ML

Corrected some math typos in the proof of Lemma 1

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/1905.13736v4)

**Authors**: Yair Carmon, Aditi Raghunathan, Ludwig Schmidt, Percy Liang, John C. Duchi

**Abstracts**: We demonstrate, theoretically and empirically, that adversarial robustness can significantly benefit from semisupervised learning. Theoretically, we revisit the simple Gaussian model of Schmidt et al. that shows a sample complexity gap between standard and robust classification. We prove that unlabeled data bridges this gap: a simple semisupervised learning procedure (self-training) achieves high robust accuracy using the same number of labels required for achieving high standard accuracy. Empirically, we augment CIFAR-10 with 500K unlabeled images sourced from 80 Million Tiny Images and use robust self-training to outperform state-of-the-art robust accuracies by over 5 points in (i) $\ell_\infty$ robustness against several strong attacks via adversarial training and (ii) certified $\ell_2$ and $\ell_\infty$ robustness via randomized smoothing. On SVHN, adding the dataset's own extra training set with the labels removed provides gains of 4 to 10 points, within 1 point of the gain from using the extra labels.

摘要: 我们从理论上和经验上证明，半监督学习可以显著提高对手的稳健性。理论上，我们重温了Schmidt等人的简单高斯模型。这显示了标准分类和健壮分类之间的样本复杂性差距。我们证明了无标签数据弥合了这一差距：一个简单的半监督学习过程(自我训练)使用相同数量的标签来实现高标准精度所需的高鲁棒精度。经验上，我们用来自8000万幅微小图像的500K未标记图像来增强CIFAR-10，并使用稳健的自我训练在(I)通过对抗性训练对几种强攻击的鲁棒性和(Ii)通过随机平滑认证的$\ell_2和$\ell_\infty$鲁棒性方面超过最先进的鲁棒准确率5个点。在SVHN上，添加删除了标签的数据集自己的额外训练集可以提供4到10个点的增益，与使用额外标签的增益相差1个点。



## **2. Attention-Guided Black-box Adversarial Attacks with Large-Scale Multiobjective Evolutionary Optimization**

基于大规模多目标进化优化的注意力引导黑盒对抗攻击 cs.CV

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2101.07512v3)

**Authors**: Jie Wang, Zhaoxia Yin, Jing Jiang, Yang Du

**Abstracts**: Fooling deep neural networks (DNNs) with the black-box optimization has become a popular adversarial attack fashion, as the structural prior knowledge of DNNs is always unknown. Nevertheless, recent black-box adversarial attacks may struggle to balance their attack ability and visual quality of the generated adversarial examples (AEs) in tackling high-resolution images. In this paper, we propose an attention-guided black-box adversarial attack based on the large-scale multiobjective evolutionary optimization, termed as LMOA. By considering the spatial semantic information of images, we firstly take advantage of the attention map to determine the perturbed pixels. Instead of attacking the entire image, reducing the perturbed pixels with the attention mechanism can help to avoid the notorious curse of dimensionality and thereby improves the performance of attacking. Secondly, a large-scale multiobjective evolutionary algorithm is employed to traverse the reduced pixels in the salient region. Benefiting from its characteristics, the generated AEs have the potential to fool target DNNs while being imperceptible by the human vision. Extensive experimental results have verified the effectiveness of the proposed LMOA on the ImageNet dataset. More importantly, it is more competitive to generate high-resolution AEs with better visual quality compared with the existing black-box adversarial attacks.

摘要: 由于深度神经网络(DNNs)的结构先验知识总是未知的，用黑盒优化来愚弄DNNs已经成为一种流行的对抗性攻击方式。然而，最近的黑盒对抗性攻击可能难以平衡它们的攻击能力和生成的对抗性示例(AE)在处理高分辨率图像时的视觉质量。本文提出了一种基于大规模多目标进化优化的注意力引导的黑盒对抗攻击，称为LMOA。考虑到图像的空间语义信息，首先利用注意力图确定扰动像素。与攻击整幅图像不同的是，利用注意力机制减少扰动像素可以帮助避免臭名昭著的维数诅咒，从而提高攻击性能。其次，采用大规模多目标进化算法遍历显著区域内的缩减像素。得益于它的特性，生成的AE有可能愚弄目标DNN，同时又是人眼看不见的。大量的实验结果验证了所提出的LMOA在ImageNet数据集上的有效性。更重要的是，与现有的黑盒对抗性攻击相比，更多的是好胜能够生成视觉质量更好的高分辨率音效。



## **3. On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles**

自主车辆轨迹预测的对抗鲁棒性研究 cs.CV

11 pages, 11 figures

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2201.05057v1)

**Authors**: Qingzhao Zhang, Shengtuo Hu, Jiachen Sun, Qi Alfred Chen, Z. Morley Mao

**Abstracts**: Trajectory prediction is a critical component for autonomous vehicles (AVs) to perform safe planning and navigation. However, few studies have analyzed the adversarial robustness of trajectory prediction or investigated whether the worst-case prediction can still lead to safe planning. To bridge this gap, we study the adversarial robustness of trajectory prediction models by proposing a new adversarial attack that perturbs normal vehicle trajectories to maximize the prediction error. Our experiments on three models and three datasets show that the adversarial prediction increases the prediction error by more than 150%. Our case studies show that if an adversary drives a vehicle close to the target AV following the adversarial trajectory, the AV may make an inaccurate prediction and even make unsafe driving decisions. We also explore possible mitigation techniques via data augmentation and trajectory smoothing.

摘要: 轨迹预测是自动驾驶车辆进行安全规划和导航的重要组成部分。然而，很少有研究分析弹道预测的对抗稳健性，或研究最坏情况的预测是否仍能导致安全规划。为了弥补这一差距，我们研究了轨迹预测模型的对抗性，提出了一种新的对抗性攻击，通过扰动正常的车辆轨迹来最大化预测误差。在三个模型和三个数据集上的实验表明，对抗性预测使预测误差增加了150%以上。我们的案例研究表明，如果对手沿着敌对的轨迹驾驶车辆接近目标AV，AV可能会做出不准确的预测，甚至做出不安全的驾驶决策。我们还通过数据增强和轨迹平滑来探索可能的缓解技术。



## **4. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

学习打破深度感知散列：用例NeuralHash cs.LG

24 pages, 16 figures, 5 tables

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2111.06628v3)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.

摘要: 苹果最近公布了其深度感知散列系统NeuralHash，用于在文件上传到其iCloud服务之前检测用户设备上的儿童性虐待材料(CSAM)。公众很快就对保护用户隐私和系统的可靠性提出了批评。本文首次提出了基于NeuralHash的深度感知散列的综合实证分析。具体地说，我们表明当前的深度感知散列可能并不健壮。攻击者可以通过在图像中应用微小的改变来操纵散列值，这些改变要么是由基于梯度的方法引起的，要么是简单地通过执行标准图像转换来强制或防止散列冲突。这样的攻击让恶意行为者很容易利用检测系统：从隐藏滥用材料到陷害无辜用户，一切皆有可能。此外，使用散列值，仍然可以对存储在用户设备上的数据进行推断。在我们看来，根据我们的结果，当前形式的深度感知散列通常不能用于健壮的客户端扫描，不应该从隐私的角度使用。



## **5. Evaluation of Four Black-box Adversarial Attacks and Some Query-efficient Improvement Analysis**

四种黑盒对抗性攻击的评测及查询效率改进分析 cs.CR

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2201.05001v1)

**Authors**: Rui Wang

**Abstracts**: With the fast development of machine learning technologies, deep learning models have been deployed in almost every aspect of everyday life. However, the privacy and security of these models are threatened by adversarial attacks. Among which black-box attack is closer to reality, where limited knowledge can be acquired from the model. In this paper, we provided basic background knowledge about adversarial attack and analyzed four black-box attack algorithms: Bandits, NES, Square Attack and ZOsignSGD comprehensively. We also explored the newly proposed Square Attack method with respect to square size, hoping to improve its query efficiency.

摘要: 随着机器学习技术的快速发展，深度学习模型几乎已经部署到日常生活的方方面面。然而，这些模型的隐私和安全受到对抗性攻击的威胁。其中黑箱攻击更接近实际，可以从模型中获取有限的知识。本文介绍了对抗性攻击的基本背景知识，并对四种黑盒攻击算法：强盗算法、NES算法、Square攻击算法和ZOsignSGD算法进行了全面的分析。我们还对新提出的关于正方形大小的正方形攻击方法进行了探索，希望能提高其查询效率。



## **6. Captcha Attack: Turning Captchas Against Humanity**

验证码攻击：将验证码变成反人类的 cs.CR

Currently under submission

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2201.04014v2)

**Authors**: Mauro Conti, Luca Pajola, Pier Paolo Tricomi

**Abstracts**: Nowadays, people generate and share massive content on online platforms (e.g., social networks, blogs). In 2021, the 1.9 billion daily active Facebook users posted around 150 thousand photos every minute. Content moderators constantly monitor these online platforms to prevent the spreading of inappropriate content (e.g., hate speech, nudity images). Based on deep learning (DL) advances, Automatic Content Moderators (ACM) help human moderators handle high data volume. Despite their advantages, attackers can exploit weaknesses of DL components (e.g., preprocessing, model) to affect their performance. Therefore, an attacker can leverage such techniques to spread inappropriate content by evading ACM.   In this work, we propose CAPtcha Attack (CAPA), an adversarial technique that allows users to spread inappropriate text online by evading ACM controls. CAPA, by generating custom textual CAPTCHAs, exploits ACM's careless design implementations and internal procedures vulnerabilities. We test our attack on real-world ACM, and the results confirm the ferocity of our simple yet effective attack, reaching up to a 100% evasion success in most cases. At the same time, we demonstrate the difficulties in designing CAPA mitigations, opening new challenges in CAPTCHAs research area.

摘要: 如今，人们在在线平台(如社交网络、博客)上生成和共享大量内容。2021年，Facebook的19亿日活跃用户每分钟发布约15万张照片。内容版主不断监控这些在线平台，以防止不恰当内容(如仇恨言论、裸照)的传播。基于深度学习(DL)的进步，自动内容版主(ACM)帮助人工版主处理大量数据。尽管具有优势，攻击者仍可以利用DL组件的弱点(例如，预处理、模型)来影响其性能。因此，攻击者可以利用这些技术通过规避ACM来传播不适当的内容。在这项工作中，我们提出了验证码攻击(CAPA)，这是一种敌意技术，允许用户通过逃避ACM控制来在线传播不恰当的文本。通过生成自定义文本验证码，CAPA可以利用ACM粗心的设计实现和内部过程漏洞。我们在真实的ACM上测试了我们的攻击，结果证实了我们简单而有效的攻击的凶猛程度，在大多数情况下达到了100%的规避成功。同时，我们论证了设计CAPA缓解措施的困难，为CAPTCHAS研究领域带来了新的挑战。



## **7. Reconstructing Training Data with Informed Adversaries**

利用知情对手重建训练数据 cs.CR

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2201.04845v1)

**Authors**: Borja Balle, Giovanni Cherubin, Jamie Hayes

**Abstracts**: Given access to a machine learning model, can an adversary reconstruct the model's training data? This work studies this question from the lens of a powerful informed adversary who knows all the training data points except one. By instantiating concrete attacks, we show it is feasible to reconstruct the remaining data point in this stringent threat model. For convex models (e.g. logistic regression), reconstruction attacks are simple and can be derived in closed-form. For more general models (e.g. neural networks), we propose an attack strategy based on training a reconstructor network that receives as input the weights of the model under attack and produces as output the target data point. We demonstrate the effectiveness of our attack on image classifiers trained on MNIST and CIFAR-10, and systematically investigate which factors of standard machine learning pipelines affect reconstruction success. Finally, we theoretically investigate what amount of differential privacy suffices to mitigate reconstruction attacks by informed adversaries. Our work provides an effective reconstruction attack that model developers can use to assess memorization of individual points in general settings beyond those considered in previous works (e.g. generative language models or access to training gradients); it shows that standard models have the capacity to store enough information to enable high-fidelity reconstruction of training data points; and it demonstrates that differential privacy can successfully mitigate such attacks in a parameter regime where utility degradation is minimal.

摘要: 如果可以访问机器学习模型，对手可以重建模型的训练数据吗？这项工作从一个强大的、消息灵通的对手的角度来研究这个问题，他知道除一个以外的所有训练数据点。通过对具体攻击的实例化，我们证明了在这种严格的威胁模型中重构剩余数据点是可行的。对于凸模型(例如Logistic回归)，重构攻击是简单的，并且可以以封闭形式导出。对于更一般的模型(如神经网络)，我们提出了一种基于训练重构器网络的攻击策略，该重构器网络接收被攻击模型的权重作为输入，并产生目标数据点作为输出。我们在MNIST和CIFAR-10上训练的图像分类器上验证了我们的攻击的有效性，并系统地研究了标准机器学习管道中哪些因素影响重建成功。最后，我们从理论上研究了多少差异隐私足以减轻知情攻击者的重构攻击。我们的工作提供了一个有效的重建攻击，模型开发者可以用它来评估在一般环境下对单个点的记忆，而不是以前的工作中考虑的那些(例如，生成性语言模型或对训练梯度的访问)；它表明标准模型有能力存储足够的信息来实现训练数据点的高保真重建；并且它证明了在效用降级最小的参数机制中，差分隐私可以成功地缓解这样的攻击。



## **8. Towards Adversarially Robust Deep Image Denoising**

面向对抗性鲁棒的深部图像去噪研究 eess.IV

**SubmitDate**: 2022-01-13    [paper-pdf](http://arxiv.org/pdf/2201.04397v2)

**Authors**: Hanshu Yan, Jingfeng Zhang, Jiashi Feng, Masashi Sugiyama, Vincent Y. F. Tan

**Abstracts**: This work systematically investigates the adversarial robustness of deep image denoisers (DIDs), i.e, how well DIDs can recover the ground truth from noisy observations degraded by adversarial perturbations. Firstly, to evaluate DIDs' robustness, we propose a novel adversarial attack, namely Observation-based Zero-mean Attack ({\sc ObsAtk}), to craft adversarial zero-mean perturbations on given noisy images. We find that existing DIDs are vulnerable to the adversarial noise generated by {\sc ObsAtk}. Secondly, to robustify DIDs, we propose an adversarial training strategy, hybrid adversarial training ({\sc HAT}), that jointly trains DIDs with adversarial and non-adversarial noisy data to ensure that the reconstruction quality is high and the denoisers around non-adversarial data are locally smooth. The resultant DIDs can effectively remove various types of synthetic and adversarial noise. We also uncover that the robustness of DIDs benefits their generalization capability on unseen real-world noise. Indeed, {\sc HAT}-trained DIDs can recover high-quality clean images from real-world noise even without training on real noisy data. Extensive experiments on benchmark datasets, including Set68, PolyU, and SIDD, corroborate the effectiveness of {\sc ObsAtk} and {\sc HAT}.

摘要: 本文系统地研究了深层图像去噪器(DIDS)的对抗鲁棒性，即DIDS从被对抗扰动降级的噪声观测中恢复地面真实信息的能力。首先，为了评估DIDS的鲁棒性，我们提出了一种新的敌意攻击方法，即基于观测的零均值攻击({\sc ObsAtk})，对给定的噪声图像进行敌意零均值扰动。我们发现现有的DID容易受到{\sc ObsAtk}产生的对抗性噪声的影响。其次，为了增强DIDs的鲁棒性，我们提出了一种对抗性训练策略--混合对抗性训练({\sc HAT})，用对抗性和非对抗性噪声数据联合训练DIDs，以确保重建质量高，非对抗性数据周围的去噪器局部光滑。由此产生的DID可以有效地去除各种类型的合成噪声和对抗性噪声。我们还发现，DID的健壮性有利于它们对不可见的真实世界噪声的泛化能力。事实上，经过{\sc HAT}训练的DID可以从真实世界的噪声中恢复高质量的干净图像，即使不需要对真实的噪声数据进行培训。在包括Set68、PolyU和SIDD在内的基准数据集上的大量实验证实了{\sc ObsAtk}和{\sc HAT}的有效性。



## **9. Security for Machine Learning-based Software Systems: a survey of threats, practices and challenges**

基于机器学习的软件系统安全：威胁、实践和挑战综述 cs.CR

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2201.04736v1)

**Authors**: Huaming Chen, M. Ali Babar

**Abstracts**: The rapid development of Machine Learning (ML) has demonstrated superior performance in many areas, such as computer vision, video and speech recognition. It has now been increasingly leveraged in software systems to automate the core tasks. However, how to securely develop the machine learning-based modern software systems (MLBSS) remains a big challenge, for which the insufficient consideration will largely limit its application in safety-critical domains. One concern is that the present MLBSS development tends to be rush, and the latent vulnerabilities and privacy issues exposed to external users and attackers will be largely neglected and hard to be identified. Additionally, machine learning-based software systems exhibit different liabilities towards novel vulnerabilities at different development stages from requirement analysis to system maintenance, due to its inherent limitations from the model and data and the external adversary capabilities. In this work, we consider that security for machine learning-based software systems may arise by inherent system defects or external adversarial attacks, and the secure development practices should be taken throughout the whole lifecycle. While machine learning has become a new threat domain for existing software engineering practices, there is no such review work covering the topic. Overall, we present a holistic review regarding the security for MLBSS, which covers a systematic understanding from a structure review of three distinct aspects in terms of security threats. Moreover, it provides a thorough state-of-the-practice for MLBSS secure development. Finally, we summarise the literature for system security assurance, and motivate the future research directions with open challenges. We anticipate this work provides sufficient discussion and novel insights to incorporate system security engineering for future exploration.

摘要: 机器学习(ML)的快速发展在计算机视觉、视频和语音识别等领域表现出了优异的性能。现在，在软件系统中越来越多地利用它来自动执行核心任务。然而，如何安全地开发基于机器学习的现代软件系统(MLBSS)仍然是一个巨大的挑战，考虑不足将在很大程度上限制其在安全关键领域的应用。一个令人担忧的问题是，目前MLBSS的开发趋于仓促，暴露给外部用户和攻击者的潜在漏洞和隐私问题将在很大程度上被忽视，难以识别。此外，基于机器学习的软件系统在从需求分析到系统维护的不同开发阶段，由于其模型和数据的固有局限性以及外部对手能力的限制，对新的漏洞表现出不同的易感性。在这项工作中，我们认为基于机器学习的软件系统的安全性可能是由固有的系统缺陷或外部的对抗性攻击引起的，安全的开发实践应该贯穿于整个生命周期。虽然机器学习已经成为现有软件工程实践的一个新的威胁领域，但还没有覆盖这个主题的这样的审查工作。总体而言，我们提出了关于MLBSS安全的整体审查，其中包括从安全威胁的三个不同方面的结构审查来系统地理解MLBSS的安全。为MLBSS的安全开发提供了全面的实践基础。最后，对系统安全保障方面的文献进行了总结，并对未来的研究方向进行了展望。我们期望这项工作能为将来的探索提供充分的讨论和新的见解，将系统安全工程纳入其中。



## **10. Adversarially Robust Classification by Conditional Generative Model Inversion**

基于条件生成模型反演的对抗性鲁棒分类 cs.LG

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2201.04733v1)

**Authors**: Mitra Alirezaei, Tolga Tasdizen

**Abstracts**: Most adversarial attack defense methods rely on obfuscating gradients. These methods are successful in defending against gradient-based attacks; however, they are easily circumvented by attacks which either do not use the gradient or by attacks which approximate and use the corrected gradient. Defenses that do not obfuscate gradients such as adversarial training exist, but these approaches generally make assumptions about the attack such as its magnitude. We propose a classification model that does not obfuscate gradients and is robust by construction without assuming prior knowledge about the attack. Our method casts classification as an optimization problem where we "invert" a conditional generator trained on unperturbed, natural images to find the class that generates the closest sample to the query image. We hypothesize that a potential source of brittleness against adversarial attacks is the high-to-low-dimensional nature of feed-forward classifiers which allows an adversary to find small perturbations in the input space that lead to large changes in the output space. On the other hand, a generative model is typically a low-to-high-dimensional mapping. While the method is related to Defense-GAN, the use of a conditional generative model and inversion in our model instead of the feed-forward classifier is a critical difference. Unlike Defense-GAN, which was shown to generate obfuscated gradients that are easily circumvented, we show that our method does not obfuscate gradients. We demonstrate that our model is extremely robust against black-box attacks and has improved robustness against white-box attacks compared to naturally trained, feed-forward classifiers.

摘要: 大多数对抗性攻击防御方法都依赖于模糊梯度。这些方法在防御基于梯度的攻击方面是成功的；但是，它们很容易被不使用梯度的攻击或近似并使用校正的梯度的攻击所绕过。存在不会混淆诸如对抗性训练等梯度的防御措施，但这些方法通常会对攻击进行假设，例如其强度。我们提出了一种分类模型，它不会混淆梯度，并且在不假设攻击先验知识的情况下，通过构造具有健壮性。我们的方法将分类转换为一个优化问题，在这个问题中，我们“反转”在未受干扰的自然图像上训练的条件生成器，以找到生成与查询图像最接近的样本的类。我们假设一个潜在的抵抗敌意攻击的脆性来源是前馈分类器从高维到低维的性质，它允许敌手在输入空间中发现导致输出空间大变化的小扰动。另一方面，生成模型通常是低维到高维的映射。虽然该方法与防御性GaN有关，但在我们的模型中使用条件生成模型和反演来代替前馈分类器是一个关键的区别。与Defense-GAN不同的是，它可以生成容易规避的模糊梯度，而我们的方法不会模糊梯度。我们证明，与自然训练的前馈分类器相比，我们的模型对黑盒攻击具有极强的鲁棒性，并且对白盒攻击具有更好的鲁棒性。



## **11. Get your Foes Fooled: Proximal Gradient Split Learning for Defense against Model Inversion Attacks on IoMT data**

愚弄你的敌人：用于防御IoMT数据模型反转攻击的近梯度分裂学习 cs.CR

9 pages, 5 figures, 2 tables

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2201.04569v1)

**Authors**: Sunder Ali Khowaja, Ik Hyun Lee, Kapal Dev, Muhammad Aslam Jarwar, Nawab Muhammad Faseeh Qureshi

**Abstracts**: The past decade has seen a rapid adoption of Artificial Intelligence (AI), specifically the deep learning networks, in Internet of Medical Things (IoMT) ecosystem. However, it has been shown recently that the deep learning networks can be exploited by adversarial attacks that not only make IoMT vulnerable to the data theft but also to the manipulation of medical diagnosis. The existing studies consider adding noise to the raw IoMT data or model parameters which not only reduces the overall performance concerning medical inferences but also is ineffective to the likes of deep leakage from gradients method. In this work, we propose proximal gradient split learning (PSGL) method for defense against the model inversion attacks. The proposed method intentionally attacks the IoMT data when undergoing the deep neural network training process at client side. We propose the use of proximal gradient method to recover gradient maps and a decision-level fusion strategy to improve the recognition performance. Extensive analysis show that the PGSL not only provides effective defense mechanism against the model inversion attacks but also helps in improving the recognition performance on publicly available datasets. We report 17.9$\%$ and 36.9$\%$ gains in accuracy over reconstructed and adversarial attacked images, respectively.

摘要: 在过去的十年中，人工智能(AI)，特别是深度学习网络，在医疗物联网(IoMT)生态系统中得到了迅速的采用。然而，最近的研究表明，深度学习网络可以被敌意攻击所利用，这些攻击不仅使物联网容易受到数据窃取的攻击，而且还容易受到医疗诊断的篡改。已有的研究都是考虑在原始IoMT数据或模型参数中加入噪声，这不仅降低了医学推断的整体性能，而且对梯度法等深度渗漏方法效果不佳。在这项工作中，我们提出了近邻梯度分裂学习(PSGL)方法来防御模型反转攻击。该方法在客户端对IoMT数据进行深度神经网络训练时，对IoMT数据进行故意攻击。提出了利用近邻梯度法恢复梯度图，并提出了决策层融合策略来提高识别性能。大量分析表明，PGSL不仅提供了对模型反转攻击的有效防御机制，而且有助于提高对公开数据集的识别性能。我们报告的准确率分别比重建图像和对抗性攻击图像提高了17.9美元和36.9美元。



## **12. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

谁是最强大的敌人？基于Deep RL的最优高效规避攻击研究 cs.LG

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2106.05087v3)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named "actor" and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.

摘要: 评估强化学习(RL)Agent在状态观测(在一定约束范围内)的最强/最优对抗扰动下的最坏情况下的性能，对于理解RL Agent的鲁棒性是至关重要的。然而，无论是从我们是否能找到最佳攻击，还是从我们找到最佳攻击的效率来看，找到最佳对手都是具有挑战性的。现有的对抗性RL研究要么使用基于启发式的方法，可能找不到最强的对手，要么将Agent视为环境的一部分，直接训练基于RL的对手，这可以找到最优的对手，但在大的状态空间中可能会变得难以处理。本文提出了一种新的攻击方法，通过设计一个名为“参与者”的函数和一个名为“导演”的基于RL的学习器之间的协作来寻找最优攻击。参与者为给定的政策扰动方向制作状态扰动，导演学习提出最佳政策扰动方向。我们提出的算法PA-AD在理论上是最优的，并且在具有大状态空间的环境中比以前的基于RL的工作效率要高得多。实验结果表明，在不同的Atari和MuJoCo环境下，我们提出的PA-AD攻击方法普遍优于最新的攻击方法。通过将PA-AD应用于对抗性训练，我们在强对手下的多任务中获得了最先进的经验鲁棒性。



## **13. Complete Traceability Multimedia Fingerprinting Codes Resistant to Averaging Attack and Adversarial Noise with Optimal Rate**

具有最佳速率的抗平均攻击和对抗噪声的完全可追溯性多媒体指纹码 cs.IT

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2108.09015v3)

**Authors**: Ilya Vorobyev

**Abstracts**: In this paper we consider complete traceability multimedia fingerprinting codes resistant to averaging attacks and adversarial noise. Recently it was shown that there are no such codes for the case of an arbitrary linear attack. However, for the case of averaging attacks complete traceability multimedia fingerprinting codes of exponential cardinality resistant to constant adversarial noise were constructed in 2020 by Egorova et al. We continue this work and provide an improved lower bound on the rate of these codes.

摘要: 在本文中，我们考虑了完全可追溯性多媒体指纹码，它能抵抗平均攻击和对抗噪声。最近有研究表明，在任意线性攻击的情况下，不存在这样的码。然而，对于平均攻击的情况，Egorova等人于2020年构造了指数基数抵抗恒定对抗噪声的完全可追溯性多媒体指纹码。我们继续这项工作，并提供了这些码率的一个改进的下界。



## **14. Game Theory for Adversarial Attacks and Defenses**

对抗性攻防的博弈论 cs.LG

With the agreement of my coauthors, I would like to withdraw the  manuscript "Game Theory for Adversarial Attacks and Defenses". Some  experimental procedures were not included in the manuscript, which makes a  part of important claims not meaningful

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2110.06166v3)

**Authors**: Shorya Sharma

**Abstracts**: Adversarial attacks can generate adversarial inputs by applying small but intentionally worst-case perturbations to samples from the dataset, which leads to even state-of-the-art deep neural networks outputting incorrect answers with high confidence. Hence, some adversarial defense techniques are developed to improve the security and robustness of the models and avoid them being attacked. Gradually, a game-like competition between attackers and defenders formed, in which both players would attempt to play their best strategies against each other while maximizing their own payoffs. To solve the game, each player would choose an optimal strategy against the opponent based on the prediction of the opponent's strategy choice. In this work, we are on the defensive side to apply game-theoretic approaches on defending against attacks. We use two randomization methods, random initialization and stochastic activation pruning, to create diversity of networks. Furthermore, we use one denoising technique, super resolution, to improve models' robustness by preprocessing images before attacks. Our experimental results indicate that those three methods can effectively improve the robustness of deep-learning neural networks.

摘要: 对抗性攻击可以通过对数据集中的样本应用小但有意的最坏情况扰动来生成对抗性输入，这甚至导致最先进的深度神经网络以高置信度输出不正确的答案。因此，为了提高模型的安全性和健壮性，避免模型受到攻击，一些对抗性防御技术应运而生。逐渐地，攻击者和后卫之间形成了一种游戏般的竞争，双方都会试图在最大化自己收益的同时，发挥自己最好的策略。为了解决博弈，每个玩家都会根据对手的策略选择预测来选择一个最优策略来对抗对手。在这项工作中，我们处于守势，应用博弈论的方法来防御攻击。我们使用随机初始化和随机激活剪枝两种随机化方法来创建网络多样性。此外，我们还使用了超分辨率去噪技术，通过在攻击前对图像进行预处理来提高模型的鲁棒性。实验结果表明，这三种方法均能有效提高深度学习神经网络的鲁棒性。



## **15. Similarity-based Gray-box Adversarial Attack Against Deep Face Recognition**

基于相似度的灰度盒对抗深度人脸识别攻击 cs.CV

ACCEPTED in IEEE International Conference on Automatic Face and  Gesture Recognition (FG 2021)

**SubmitDate**: 2022-01-12    [paper-pdf](http://arxiv.org/pdf/2201.04011v2)

**Authors**: Hanrui Wang, Shuo Wang, Zhe Jin, Yandan Wang, Cunjian Chen, Massimo Tistarell

**Abstracts**: The majority of adversarial attack techniques perform well against deep face recognition when the full knowledge of the system is revealed (\emph{white-box}). However, such techniques act unsuccessfully in the gray-box setting where the face templates are unknown to the attackers. In this work, we propose a similarity-based gray-box adversarial attack (SGADV) technique with a newly developed objective function. SGADV utilizes the dissimilarity score to produce the optimized adversarial example, i.e., similarity-based adversarial attack. This technique applies to both white-box and gray-box attacks against authentication systems that determine genuine or imposter users using the dissimilarity score. To validate the effectiveness of SGADV, we conduct extensive experiments on face datasets of LFW, CelebA, and CelebA-HQ against deep face recognition models of FaceNet and InsightFace in both white-box and gray-box settings. The results suggest that the proposed method significantly outperforms the existing adversarial attack techniques in the gray-box setting. We hence summarize that the similarity-base approaches to develop the adversarial example could satisfactorily cater to the gray-box attack scenarios for de-authentication.

摘要: 当系统的全部知识被揭示时，大多数对抗性攻击技术对深度人脸识别表现良好(\emph{白盒})。然而，这样的技术在人脸模板对攻击者未知的灰盒设置中不起作用。在这项工作中，我们提出了一种基于相似度的灰盒对抗攻击(SGADV)技术，并引入了一个新的目标函数。SGADV利用相异度来产生优化的对抗性实例，即基于相似度的对抗性攻击。该技术既适用于针对身份验证系统的白盒和灰盒攻击，也适用于使用相异分数来确定真实用户或冒牌用户的身份验证系统。为了验证SGADV的有效性，我们在LFW、CelebA和CelebA-HQ的人脸数据集上与FaceNet和InsightFace的深度人脸识别模型在白盒和灰盒环境下进行了广泛的实验。结果表明，在灰箱环境下，该方法的性能明显优于现有的对抗性攻击技术。因此，我们总结出基于相似性的方法来开发敌意示例可以令人满意地迎合取消认证的灰盒攻击场景。



## **16. Tor circuit fingerprinting defenses using adaptive padding**

使用自适应填充的ToR电路指纹防御 cs.CR

17 pages

**SubmitDate**: 2022-01-11    [paper-pdf](http://arxiv.org/pdf/2103.03831v2)

**Authors**: George Kadianakis, Theodoros Polyzos, Mike Perry, Kostas Chatzikokolakis

**Abstracts**: Online anonymity and privacy has been based on confusing the adversary by creating indistinguishable network elements. Tor is the largest and most widely deployed anonymity system, designed against realistic modern adversaries. Recently, researchers have managed to fingerprint Tor's circuits -- and hence the type of underlying traffic -- simply by capturing and analyzing traffic traces. In this work, we study the circuit fingerprinting problem, isolating it from website fingerprinting, and revisit previous findings in this model, showing that accurate attacks are possible even when the application-layer traffic is identical. We then proceed to incrementally create defenses against circuit fingerprinting, using a generic adaptive padding framework for Tor based on WTF-PAD. We present a simple defense which delays a fraction of the traffic, as well as a more advanced one which can effectively hide onion service circuits with zero delays. We thoroughly evaluate both defenses, both analytically and experimentally, discovering new subtle fingerprints, but also showing the effectiveness of our defenses.

摘要: 在线匿名和隐私一直基于通过创建难以区分的网络元素来迷惑对手。Tor是最大和部署最广泛的匿名系统，专为现实的现代对手而设计。最近，研究人员仅通过捕捉和分析交通痕迹就成功地提取了Tor电路的指纹--因此也就是潜在的交通类型。在这项工作中，我们研究了电路指纹问题，将其从网站指纹中分离出来，并重新审视了该模型中以前的发现，表明即使在应用层流量相同的情况下，准确的攻击也是可能的。然后，我们使用基于WTF-PAD的Tor通用自适应填充框架，逐步创建对电路指纹的防御。我们提出了一个简单的防御方案，它可以延迟一小部分流量，以及一个更高级的防御方案，它可以有效地隐藏洋葱服务电路，而不会造成延迟。我们通过分析和实验对这两种防御措施进行了彻底评估，发现了新的微妙指纹，但也显示了我们防御措施的有效性。



## **17. An Empirical Assessment of Endpoint Security Systems Against Advanced Persistent Threats Attack Vectors**

终端安全系统抵御高级持续威胁攻击矢量的经验评估 cs.CR

This is the revised (and final) version of  https://doi.org/10.3390/jcp1030021 with more EDRs and proper classification  of products into EDRs and EPPs

**SubmitDate**: 2022-01-11    [paper-pdf](http://arxiv.org/pdf/2108.10422v2)

**Authors**: George Karantzas, Constantinos Patsakis

**Abstracts**: Advanced persistent threats pose a significant challenge for blue teams as they apply various attacks over prolonged periods, impeding event correlation and their detection. In this work, we leverage various diverse attack scenarios to assess the efficacy of EDRs and other endpoint security solutions against detecting and preventing APTs. Our results indicate that there is still a lot of room for improvement as state of the art endpoint security systems fail to prevent and log the bulk of the attacks that are reported in this work. Additionally, we discuss methods to tamper with the telemetry providers of EDRs, allowing an adversary to perform a more stealth attack.

摘要: 高级持续威胁对蓝色团队构成重大挑战，因为他们长时间应用各种攻击，阻碍事件关联及其检测。在这项工作中，我们利用各种不同的攻击场景来评估EDR和其他端点安全解决方案在检测和预防APT方面的有效性。我们的结果表明仍然有很大的改进空间，因为最先进的端点安全系统无法预防和记录本工作中报告的大量攻击。此外，我们还讨论了篡改EDR遥测提供商的方法，允许对手执行更隐蔽的攻击。



## **18. Quantifying Robustness to Adversarial Word Substitutions**

量化对敌意单词替换的健壮性 cs.CL

**SubmitDate**: 2022-01-11    [paper-pdf](http://arxiv.org/pdf/2201.03829v1)

**Authors**: Yuting Yang, Pei Huang, FeiFei Ma, Juan Cao, Meishan Zhang, Jian Zhang, Jintao Li

**Abstracts**: Deep-learning-based NLP models are found to be vulnerable to word substitution perturbations. Before they are widely adopted, the fundamental issues of robustness need to be addressed. Along this line, we propose a formal framework to evaluate word-level robustness. First, to study safe regions for a model, we introduce robustness radius which is the boundary where the model can resist any perturbation. As calculating the maximum robustness radius is computationally hard, we estimate its upper and lower bound. We repurpose attack methods as ways of seeking upper bound and design a pseudo-dynamic programming algorithm for a tighter upper bound. Then verification method is utilized for a lower bound. Further, for evaluating the robustness of regions outside a safe radius, we reexamine robustness from another view: quantification. A robustness metric with a rigorous statistical guarantee is introduced to measure the quantification of adversarial examples, which indicates the model's susceptibility to perturbations outside the safe radius. The metric helps us figure out why state-of-the-art models like BERT can be easily fooled by a few word substitutions, but generalize well in the presence of real-world noises.

摘要: 基于深度学习的自然语言处理模型容易受到单词替换扰动的影响。在它们被广泛采用之前，需要解决健壮性的基本问题。沿着这一思路，我们提出了一个评估词级健壮性的形式化框架。首先，为了研究模型的安全域，我们引入了鲁棒半径，即模型可以抵抗任何扰动的边界。由于计算最大鲁棒性半径比较困难，我们估计了它的上下界。我们将攻击方法重新定位为寻找上界的方法，并设计了一种伪动态规划算法来获得更紧的上界。然后利用验证方法确定下限。此外，为了评估安全半径以外区域的稳健性，我们从另一个角度重新检查了稳健性：量化。引入了一种具有严格统计保证的稳健性度量来度量对抗性实例的量化，该度量表明了模型对安全半径以外的扰动的敏感性。这一指标帮助我们弄清楚，为什么像伯特这样的最先进的模型很容易被几个词的替换所愚弄，但在现实世界的噪音面前却能很好地泛化。



## **19. Attacking Video Recognition Models with Bullet-Screen Comments**

用弹幕评论攻击视频识别模型 cs.CV

**SubmitDate**: 2022-01-11    [paper-pdf](http://arxiv.org/pdf/2110.15629v2)

**Authors**: Kai Chen, Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Recent research has demonstrated that Deep Neural Networks (DNNs) are vulnerable to adversarial patches which introduce perceptible but localized changes to the input. Nevertheless, existing approaches have focused on generating adversarial patches on images, their counterparts in videos have been less explored. Compared with images, attacking videos is much more challenging as it needs to consider not only spatial cues but also temporal cues. To close this gap, we introduce a novel adversarial attack in this paper, the bullet-screen comment (BSC) attack, which attacks video recognition models with BSCs. Specifically, adversarial BSCs are generated with a Reinforcement Learning (RL) framework, where the environment is set as the target model and the agent plays the role of selecting the position and transparency of each BSC. By continuously querying the target models and receiving feedback, the agent gradually adjusts its selection strategies in order to achieve a high fooling rate with non-overlapping BSCs. As BSCs can be regarded as a kind of meaningful patch, adding it to a clean video will not affect people' s understanding of the video content, nor will arouse people' s suspicion. We conduct extensive experiments to verify the effectiveness of the proposed method. On both UCF-101 and HMDB-51 datasets, our BSC attack method can achieve about 90\% fooling rate when attacking three mainstream video recognition models, while only occluding \textless 8\% areas in the video. Our code is available at https://github.com/kay-ck/BSC-attack.

摘要: 最近的研究表明，深度神经网络(DNNs)很容易受到敌意补丁的攻击，这些补丁会给输入带来可感知的局部变化。然而，现有的方法主要集中在生成图像上的对抗性补丁，而对视频中的对应补丁的研究较少。与图像相比，攻击视频更具挑战性，因为它不仅需要考虑空间线索，还需要考虑时间线索。为了缩小这一差距，本文引入了一种新的对抗性攻击，即弹幕评论(BSC)攻击，它利用弹幕评论攻击视频识别模型。具体地说，利用强化学习(RL)框架生成对抗性BSC，其中环境被设置为目标模型，Agent扮演选择每个BSC的位置和透明度的角色。通过不断查询目标模型并接收反馈，Agent逐渐调整其选择策略，以获得不重叠的BSC的较高愚弄率。由于BSCS可以看作是一种有意义的补丁，将其添加到干净的视频中不会影响人们对视频内容的理解，也不会引起人们的怀疑。为了验证该方法的有效性，我们进行了大量的实验。在UCF-101和HMDB-51两个数据集上，我们的BSC攻击方法在攻击三种主流视频识别模型时，仅对视频中的8个无遮挡区域进行攻击，可以达到约90%的愚弄率。我们的代码可在https://github.com/kay-ck/BSC-attack.获得



## **20. Sequential Randomized Smoothing for Adversarially Robust Speech Recognition**

对抗性语音识别的序贯随机平滑算法 cs.CL

This update adds some relevant references to past and concurrent work

**SubmitDate**: 2022-01-10    [paper-pdf](http://arxiv.org/pdf/2112.03000v2)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: While Automatic Speech Recognition has been shown to be vulnerable to adversarial attacks, defenses against these attacks are still lagging. Existing, naive defenses can be partially broken with an adaptive attack. In classification tasks, the Randomized Smoothing paradigm has been shown to be effective at defending models. However, it is difficult to apply this paradigm to ASR tasks, due to their complexity and the sequential nature of their outputs. Our paper overcomes some of these challenges by leveraging speech-specific tools like enhancement and ROVER voting to design an ASR model that is robust to perturbations. We apply adaptive versions of state-of-the-art attacks, such as the Imperceptible ASR attack, to our model, and show that our strongest defense is robust to all attacks that use inaudible noise, and can only be broken with very high distortion.

摘要: 虽然自动语音识别已经被证明容易受到对手的攻击，但对这些攻击的防御仍然滞后。现有的天真防御可以通过适应性攻击部分被打破。在分类任务中，随机平滑范例已被证明在保护模型方面是有效的。然而，由于ASR任务的复杂性及其输出的顺序性，很难将此范例应用于ASR任务。我们的论文通过利用语音专用工具(如增强和漫游投票)来设计一个对扰动具有健壮性的ASR模型，从而克服了其中的一些挑战。我们将最先进的攻击的自适应版本(如不可感知的ASR攻击)应用到我们的模型中，并表明我们最强的防御对所有使用不可听噪声的攻击都是健壮的，并且只有在非常高的失真情况下才能破解。



## **21. GMFIM: A Generative Mask-guided Facial Image Manipulation Model for Privacy Preservation**

GMFIM：一种产生式面具引导的隐私保护人脸图像处理模型 cs.CV

**SubmitDate**: 2022-01-10    [paper-pdf](http://arxiv.org/pdf/2201.03353v1)

**Authors**: Mohammad Hossein Khojaste, Nastaran Moradzadeh Farid, Ahmad Nickabadi

**Abstracts**: The use of social media websites and applications has become very popular and people share their photos on these networks. Automatic recognition and tagging of people's photos on these networks has raised privacy preservation issues and users seek methods for hiding their identities from these algorithms. Generative adversarial networks (GANs) are shown to be very powerful in generating face images in high diversity and also in editing face images. In this paper, we propose a Generative Mask-guided Face Image Manipulation (GMFIM) model based on GANs to apply imperceptible editing to the input face image to preserve the privacy of the person in the image. Our model consists of three main components: a) the face mask module to cut the face area out of the input image and omit the background, b) the GAN-based optimization module for manipulating the face image and hiding the identity and, c) the merge module for combining the background of the input image and the manipulated de-identified face image. Different criteria are considered in the loss function of the optimization step to produce high-quality images that are as similar as possible to the input image while they cannot be recognized by AFR systems. The results of the experiments on different datasets show that our model can achieve better performance against automated face recognition systems in comparison to the state-of-the-art methods and it catches a higher attack success rate in most experiments from a total of 18. Moreover, the generated images of our proposed model have the highest quality and are more pleasing to human eyes.

摘要: 社交媒体网站和应用程序的使用已经变得非常流行，人们在这些网络上分享他们的照片。自动识别和标记人们在这些网络上的照片引发了隐私保护问题，用户寻求对这些算法隐藏身份的方法。生成性对抗网络(GAN)在生成高度多样化的人脸图像和编辑人脸图像方面表现出非常强大的能力。本文提出了一种基于遗传算法的产生式面具引导人脸图像处理(GMFIM)模型，对输入的人脸图像进行不可感知编辑，以保护图像中人的隐私。该模型由三个主要部分组成：a)人脸模板模块，用于从输入图像中分割出人脸区域并省略背景；b)基于GAN的优化模块，用于操作人脸图像并隐藏身份；c)合并模块，用于合并输入图像的背景和处理后的去身份人脸图像。在优化步骤的损失函数中考虑了不同的准则，以产生与输入图像尽可能相似的高质量图像，而这些图像又不能被AFR系统识别。在不同数据集上的实验结果表明，与目前最先进的方法相比，该模型在自动人脸识别系统中取得了更好的性能，并且在大多数实验中获得了更高的攻击成功率(总共18个)，并且生成的图像质量最高，更符合人眼的口味。



## **22. Evaluation of Neural Networks Defenses and Attacks using NDCG and Reciprocal Rank Metrics**

基于NDCG和倒数秩度量的神经网络防御与攻击评估 cs.CR

12 pages, 5 figures

**SubmitDate**: 2022-01-10    [paper-pdf](http://arxiv.org/pdf/2201.05071v1)

**Authors**: Haya Brama, Lihi Dery, Tal Grinshpoun

**Abstracts**: The problem of attacks on neural networks through input modification (i.e., adversarial examples) has attracted much attention recently. Being relatively easy to generate and hard to detect, these attacks pose a security breach that many suggested defenses try to mitigate. However, the evaluation of the effect of attacks and defenses commonly relies on traditional classification metrics, without adequate adaptation to adversarial scenarios. Most of these metrics are accuracy-based, and therefore may have a limited scope and low distinctive power. Other metrics do not consider the unique characteristics of neural networks functionality, or measure the effect of the attacks indirectly (e.g., through the complexity of their generation). In this paper, we present two metrics which are specifically designed to measure the effect of attacks, or the recovery effect of defenses, on the output of neural networks in multiclass classification tasks. Inspired by the normalized discounted cumulative gain and the reciprocal rank metrics used in information retrieval literature, we treat the neural network predictions as ranked lists of results. Using additional information about the probability of the rank enabled us to define novel metrics that are suited to the task at hand. We evaluate our metrics using various attacks and defenses on a pretrained VGG19 model and the ImageNet dataset. Compared to the common classification metrics, our proposed metrics demonstrate superior informativeness and distinctiveness.

摘要: 近年来，通过修改输入(即对抗性例子)来攻击神经网络的问题引起了人们的广泛关注。由于相对容易生成和难以检测，这些攻击构成了安全漏洞，许多人建议防御措施试图缓解这一点。然而，对攻击和防御效果的评估通常依赖于传统的分类度量，不能很好地适应对抗性场景。这些度量中的大多数都是基于准确性的，因此可能范围有限，区分能力较低。其他度量没有考虑神经网络功能的独特特性，或者间接地测量攻击的效果(例如，通过其生成的复杂性)。在本文中，我们提出了两个专门用来度量多类分类任务中攻击或防御恢复效果对神经网络输出的影响的度量标准，这两个度量是用来衡量多类分类任务中的攻击或防御对神经网络输出的影响的。受信息检索文献中使用的归一化折现累积增益和倒数秩度量的启发，我们将神经网络预测视为结果的排序列表。使用有关排名概率的附加信息使我们能够定义适合手头任务的新度量。我们在预先训练的VGG19模型和ImageNet数据集上使用各种攻击和防御来评估我们的指标。与常用的分类度量相比，我们提出的度量具有更好的信息性和独特性。



## **23. IoTGAN: GAN Powered Camouflage Against Machine Learning Based IoT Device Identification**

IoTGAN：基于机器学习的GAN伪装技术对抗物联网设备识别 cs.CR

**SubmitDate**: 2022-01-10    [paper-pdf](http://arxiv.org/pdf/2201.03281v1)

**Authors**: Tao Hou, Tao Wang, Zhuo Lu, Yao Liu, Yalin Sagduyu

**Abstracts**: With the proliferation of IoT devices, researchers have developed a variety of IoT device identification methods with the assistance of machine learning. Nevertheless, the security of these identification methods mostly depends on collected training data. In this research, we propose a novel attack strategy named IoTGAN to manipulate an IoT device's traffic such that it can evade machine learning based IoT device identification. In the development of IoTGAN, we have two major technical challenges: (i) How to obtain the discriminative model in a black-box setting, and (ii) How to add perturbations to IoT traffic through the manipulative model, so as to evade the identification while not influencing the functionality of IoT devices. To address these challenges, a neural network based substitute model is used to fit the target model in black-box settings, it works as a discriminative model in IoTGAN. A manipulative model is trained to add adversarial perturbations into the IoT device's traffic to evade the substitute model. Experimental results show that IoTGAN can successfully achieve the attack goals. We also develop efficient countermeasures to protect machine learning based IoT device identification from been undermined by IoTGAN.

摘要: 随着物联网设备的激增，研究人员借助机器学习开发了多种物联网设备识别方法。然而，这些识别方法的安全性很大程度上依赖于收集的训练数据。在这项研究中，我们提出了一种新的攻击策略IoTGAN来操纵物联网设备的流量，使其能够逃避基于机器学习的物联网设备识别。在IoTGAN的开发中，我们面临着两大技术挑战：(I)如何在黑盒环境下获得判别模型；(Ii)如何通过操控模型对物联网流量进行扰动，从而在不影响物联网设备功能的情况下逃避识别。针对这些挑战，在黑盒环境下采用基于神经网络的替身模型对目标模型进行拟合，将其作为物联网中的判别模型。训练一个操纵性模型，将对抗性扰动添加到物联网设备的流量中，以规避替身模型。实验结果表明，IoTGAN能够成功实现攻击目标。我们还开发了有效的对策来保护基于机器学习的物联网设备识别免受IoTGAN的破坏。



## **24. Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models**

对抗性胶水：语言模型健壮性评估的多任务基准 cs.CL

Oral Presentation in NeurIPS 2021 (Datasets and Benchmarks Track). 24  pages, 4 figures, 12 tables

**SubmitDate**: 2022-01-10    [paper-pdf](http://arxiv.org/pdf/2111.02840v2)

**Authors**: Boxin Wang, Chejian Xu, Shuohang Wang, Zhe Gan, Yu Cheng, Jianfeng Gao, Ahmed Hassan Awadallah, Bo Li

**Abstracts**: Large-scale pre-trained language models have achieved tremendous success across a wide range of natural language understanding (NLU) tasks, even surpassing human performance. However, recent studies reveal that the robustness of these models can be challenged by carefully crafted textual adversarial examples. While several individual datasets have been proposed to evaluate model robustness, a principled and comprehensive benchmark is still missing. In this paper, we present Adversarial GLUE (AdvGLUE), a new multi-task benchmark to quantitatively and thoroughly explore and evaluate the vulnerabilities of modern large-scale language models under various types of adversarial attacks. In particular, we systematically apply 14 textual adversarial attack methods to GLUE tasks to construct AdvGLUE, which is further validated by humans for reliable annotations. Our findings are summarized as follows. (i) Most existing adversarial attack algorithms are prone to generating invalid or ambiguous adversarial examples, with around 90% of them either changing the original semantic meanings or misleading human annotators as well. Therefore, we perform a careful filtering process to curate a high-quality benchmark. (ii) All the language models and robust training methods we tested perform poorly on AdvGLUE, with scores lagging far behind the benign accuracy. We hope our work will motivate the development of new adversarial attacks that are more stealthy and semantic-preserving, as well as new robust language models against sophisticated adversarial attacks. AdvGLUE is available at https://adversarialglue.github.io.

摘要: 大规模的预训练语言模型在广泛的自然语言理解(NLU)任务中取得了巨大的成功，甚至超过了人类的表现。然而，最近的研究表明，这些模型的稳健性可能会受到精心设计的文本对抗性例子的挑战。虽然已经提出了几个单独的数据集来评估模型的稳健性，但仍然缺乏一个原则性和综合性的基准。本文提出了一种新的多任务基准--对抗性粘合剂(AdvGLUE)，用以定量、深入地研究和评估现代大规模语言模型在各种类型的对抗性攻击下的脆弱性。特别是，我们系统地应用了14种文本对抗性攻击方法来粘合任务来构建AdvGLUE，并进一步验证了该方法的可靠性。我们的发现总结如下。(I)现有的对抗性攻击算法大多容易产生无效或歧义的对抗性示例，其中90%左右的算法要么改变了原有的语义，要么误导了人类的注释者。因此，我们执行仔细的筛选过程来策划一个高质量的基准。(Ii)我们测试的所有语言模型和稳健训练方法在AdvGLUE上的表现都很差，分数远远落后于良性准确率。我们希望我们的工作将促进更隐蔽性和语义保持的新的对抗性攻击的发展，以及针对复杂的对抗性攻击的新的健壮语言模型的开发。有关AdvGLUE的信息，请访问https://adversarialglue.github.io.。



## **25. FLAME: Taming Backdoors in Federated Learning**

火焰：联合学习中的后门驯服 cs.CR

To appear in the 31st USENIX Security Symposium, August 2022, Boston,  MA, USA

**SubmitDate**: 2022-01-09    [paper-pdf](http://arxiv.org/pdf/2101.02281v3)

**Authors**: Thien Duc Nguyen, Phillip Rieger, Huili Chen, Hossein Yalame, Helen Möllering, Hossein Fereidooni, Samuel Marchal, Markus Miettinen, Azalia Mirhoseini, Shaza Zeitouni, Farinaz Koushanfar, Ahmad-Reza Sadeghi, Thomas Schneider

**Abstracts**: Federated Learning (FL) is a collaborative machine learning approach allowing participants to jointly train a model without having to share their private, potentially sensitive local datasets with others. Despite its benefits, FL is vulnerable to so-called backdoor attacks, in which an adversary injects manipulated model updates into the federated model aggregation process so that the resulting model will provide targeted false predictions for specific adversary-chosen inputs. Proposed defenses against backdoor attacks based on detecting and filtering out malicious model updates consider only very specific and limited attacker models, whereas defenses based on differential privacy-inspired noise injection significantly deteriorate the benign performance of the aggregated model. To address these deficiencies, we introduce FLAME, a defense framework that estimates the sufficient amount of noise to be injected to ensure the elimination of backdoors. To minimize the required amount of noise, FLAME uses a model clustering and weight clipping approach. This ensures that FLAME can maintain the benign performance of the aggregated model while effectively eliminating adversarial backdoors. Our evaluation of FLAME on several datasets stemming from application areas including image classification, word prediction, and IoT intrusion detection demonstrates that FLAME removes backdoors effectively with a negligible impact on the benign performance of the models.

摘要: 联合学习(FL)是一种协作的机器学习方法，允许参与者联合训练模型，而不必与其他人共享他们私有的、潜在敏感的局部数据集。尽管FL有好处，但它很容易受到所谓的后门攻击，即对手将操纵的模型更新注入到联合模型聚合过程中，从而产生的模型将为对手选择的特定输入提供有针对性的错误预测。已提出的基于检测和过滤恶意模型更新的后门攻击防御方案只考虑非常具体和有限的攻击者模型，而基于差分隐私激发噪声注入的防御方案会显著降低聚合模型的良性性能。为了解决这些不足，我们引入了FLAME，这是一个防御框架，可以估计要注入的足够数量的噪音，以确保消除后门。为了最大限度地减少所需的噪声量，火焰使用模型聚类和权重裁剪方法。这保证了FLAME能够在有效消除敌意后门的同时，保持聚合模型的良性性能。我们在几个来自图像分类、词语预测和物联网入侵检测等应用领域的数据集上的评估表明，FLAME有效地移除了后门，而对模型的良性性能的影响可以忽略不计。



## **26. Privacy-aware Early Detection of COVID-19 through Adversarial Training**

通过对抗性训练实现具有隐私意识的冠状病毒早期检测 cs.LG

**SubmitDate**: 2022-01-09    [paper-pdf](http://arxiv.org/pdf/2201.03004v1)

**Authors**: Omid Rohanian, Samaneh Kouchaki, Andrew Soltan, Jenny Yang, Morteza Rohanian, Yang Yang, David Clifton

**Abstracts**: Early detection of COVID-19 is an ongoing area of research that can help with triage, monitoring and general health assessment of potential patients and may reduce operational strain on hospitals that cope with the coronavirus pandemic. Different machine learning techniques have been used in the literature to detect coronavirus using routine clinical data (blood tests, and vital signs). Data breaches and information leakage when using these models can bring reputational damage and cause legal issues for hospitals. In spite of this, protecting healthcare models against leakage of potentially sensitive information is an understudied research area. In this work, we examine two machine learning approaches, intended to predict a patient's COVID-19 status using routinely collected and readily available clinical data. We employ adversarial training to explore robust deep learning architectures that protect attributes related to demographic information about the patients. The two models we examine in this work are intended to preserve sensitive information against adversarial attacks and information leakage. In a series of experiments using datasets from the Oxford University Hospitals, Bedfordshire Hospitals NHS Foundation Trust, University Hospitals Birmingham NHS Foundation Trust, and Portsmouth Hospitals University NHS Trust we train and test two neural networks that predict PCR test results using information from basic laboratory blood tests, and vital signs performed on a patients' arrival to hospital. We assess the level of privacy each one of the models can provide and show the efficacy and robustness of our proposed architectures against a comparable baseline. One of our main contributions is that we specifically target the development of effective COVID-19 detection models with built-in mechanisms in order to selectively protect sensitive attributes against adversarial attacks.

摘要: 及早发现冠状病毒是一个正在进行的研究领域，可以帮助对潜在患者进行分类、监测和一般健康评估，并可能减轻应对冠状病毒大流行的医院的运营压力。文献中已经使用了不同的机器学习技术来使用常规临床数据(血液测试和生命体征)来检测冠状病毒。使用这些模型时，数据泄露和信息泄露可能会给医院带来声誉损害和法律问题。尽管如此，保护医疗保健模型免受潜在敏感信息泄露的研究仍然是一个研究不足的领域。在这项工作中，我们检查了两种机器学习方法，旨在使用常规收集的和随时可用的临床数据来预测患者的冠状病毒状态。我们采用对抗性训练来探索健壮的深度学习架构，以保护与患者的人口统计信息相关的属性。我们在这项工作中研究的两个模型旨在保护敏感信息，使其免受对手攻击和信息泄露。在使用牛津大学医院、贝德福德郡医院NHS基金会信托基金、伯明翰大学医院NHS基金会信托基金和朴茨茅斯医院NHS基金会信托基金的数据集进行的一系列实验中，我们训练和测试了两个神经网络，它们使用来自基本实验室血液测试的信息和患者到达医院时执行的生命体征来预测PCR测试结果。我们评估每个模型可以提供的隐私级别，并根据可比较的基准显示我们建议的体系结构的有效性和健壮性。我们的主要贡献之一是，我们专门针对具有内置机制的有效冠状病毒检测模型的开发，以便选择性地保护敏感属性免受敌意攻击。



## **27. Tiny Adversarial Mulit-Objective Oneshot Neural Architecture Search**

微小对抗性多目标单目标神经结构搜索 cs.LG

**SubmitDate**: 2022-01-09    [paper-pdf](http://arxiv.org/pdf/2103.00363v2)

**Authors**: Guoyang Xie, Jinbao Wang, Guo Yu, Feng Zheng, Yaochu Jin

**Abstracts**: Due to limited computational cost and energy consumption, most neural network models deployed in mobile devices are tiny. However, tiny neural networks are commonly very vulnerable to attacks. Current research has proved that larger model size can improve robustness, but little research focuses on how to enhance the robustness of tiny neural networks. Our work focuses on how to improve the robustness of tiny neural networks without seriously deteriorating of clean accuracy under mobile-level resources. To this end, we propose a multi-objective oneshot network architecture search (NAS) algorithm to obtain the best trade-off networks in terms of the adversarial accuracy, the clean accuracy and the model size. Specifically, we design a novel search space based on new tiny blocks and channels to balance model size and adversarial performance. Moreover, since the supernet significantly affects the performance of subnets in our NAS algorithm, we reveal the insights into how the supernet helps to obtain the best subnet under white-box adversarial attacks. Concretely, we explore a new adversarial training paradigm by analyzing the adversarial transferability, the width of the supernet and the difference between training the subnets from scratch and fine-tuning. Finally, we make a statistical analysis for the layer-wise combination of certain blocks and channels on the first non-dominated front, which can serve as a guideline to design tiny neural network architectures for the resilience of adversarial perturbations.

摘要: 由于有限的计算成本和能量消耗，大多数部署在移动设备上的神经网络模型都很小。然而，微小的神经网络通常非常容易受到攻击。目前的研究已经证明，较大的模型规模可以提高鲁棒性，但很少有人关注如何提高微小神经网络的鲁棒性。我们的工作集中在如何提高微小神经网络的健壮性，而不严重降低移动级资源下的清洁精度。为此，我们提出了一种多目标OneShot网络结构搜索(NAS)算法，以获得在对抗准确率、干净准确率和模型规模方面的最佳折衷网络。具体地说，我们设计了一种新的基于新的小块和新通道的搜索空间，以平衡模型大小和对抗性能。此外，由于超网对NAS算法中子网的性能影响很大，我们揭示了在白盒攻击下，超网是如何帮助获得最优子网的。具体地说，通过分析对抗性可转换性、超网宽度以及从头开始训练子网与微调训练子网的区别，探索了一种新的对抗性训练范式。最后，我们对第一个非支配前线上某些块和信道的分层组合进行了统计分析，这对于设计具有抗对抗扰动能力的微小神经网络结构具有一定的指导意义。



## **28. Attacking Vertical Collaborative Learning System Using Adversarial Dominating Inputs**

利用对抗性主导输入攻击垂直协作学习系统 cs.CR

**SubmitDate**: 2022-01-08    [paper-pdf](http://arxiv.org/pdf/2201.02775v1)

**Authors**: Qi Pang, Yuanyuan Yuan, Shuai Wang

**Abstracts**: Vertical collaborative learning system also known as vertical federated learning (VFL) system has recently become prominent as a concept to process data distributed across many individual sources without the need to centralize it. Multiple participants collaboratively train models based on their local data in a privacy-preserving manner. To date, VFL has become a de facto solution to securely learn a model among organizations, allowing knowledge to be shared without compromising privacy of any individual organizations.   Despite the prosperous development of VFL systems, we find that certain inputs of a participant, named adversarial dominating inputs (ADIs), can dominate the joint inference towards the direction of the adversary's will and force other (victim) participants to make negligible contributions, losing rewards that are usually offered regarding the importance of their contributions in collaborative learning scenarios.   We conduct a systematic study on ADIs by first proving their existence in typical VFL systems. We then propose gradient-based methods to synthesize ADIs of various formats and exploit common VFL systems. We further launch greybox fuzz testing, guided by the resiliency score of "victim" participants, to perturb adversary-controlled inputs and systematically explore the VFL attack surface in a privacy-preserving manner. We conduct an in-depth study on the influence of critical parameters and settings in synthesizing ADIs. Our study reveals new VFL attack opportunities, promoting the identification of unknown threats before breaches and building more secure VFL systems.

摘要: 垂直协作学习系统，也称为垂直联合学习(VFL)系统，作为一种不需要集中处理分布在多个独立数据源上的数据的概念，最近已经成为一个重要的概念。多个参与者以保护隐私的方式基于其本地数据协作训练模型。到目前为止，VFL已经成为在组织之间安全地学习模型的事实上的解决方案，允许在不损害任何单个组织隐私的情况下共享知识。尽管虚拟外语系统的蓬勃发展，我们发现参与者的某些输入，称为对抗性主导输入(ADI)，可以主导朝着对手意愿方向的联合推理，并迫使其他(受害者)参与者做出可以忽略的贡献，失去通常提供的关于他们在协作学习场景中贡献的重要性的奖励。我们首先通过证明ADIs在典型的VFL系统中的存在性，对ADIs进行了系统的研究。然后，我们提出了基于梯度的方法来综合各种格式的ADI，并开发通用的VFL系统。我们进一步启动灰盒模糊测试，以“受害者”参与者的弹性分数为指导，扰乱对手控制的输入，并以保护隐私的方式系统地探索VFL攻击面。我们深入研究了关键参数和设置对ADI合成的影响。我们的研究揭示了新的VFL攻击机会，促进了在入侵之前识别未知威胁，并建立了更安全的VFL系统。



## **29. Trade-offs between membership privacy & adversarially robust learning**

会员隐私和对抗性稳健学习之间的权衡 cs.LG

**SubmitDate**: 2022-01-08    [paper-pdf](http://arxiv.org/pdf/2006.04622v2)

**Authors**: Jamie Hayes

**Abstracts**: Historically, machine learning methods have not been designed with security in mind. In turn, this has given rise to adversarial examples, carefully perturbed input samples aimed to mislead detection at test time, which have been applied to attack spam and malware classification, and more recently to attack image classification. Consequently, an abundance of research has been devoted to designing machine learning methods that are robust to adversarial examples. Unfortunately, there are desiderata besides robustness that a secure and safe machine learning model must satisfy, such as fairness and privacy. Recent work by Song et al. (2019) has shown, empirically, that there exists a trade-off between robust and private machine learning models. Models designed to be robust to adversarial examples often overfit on training data to a larger extent than standard (non-robust) models. If a dataset contains private information, then any statistical test that separates training and test data by observing a model's outputs can represent a privacy breach, and if a model overfits on training data, these statistical tests become easier.   In this work, we identify settings where standard models will overfit to a larger extent in comparison to robust models, and as empirically observed in previous works, settings where the opposite behavior occurs. Thus, it is not necessarily the case that privacy must be sacrificed to achieve robustness. The degree of overfitting naturally depends on the amount of data available for training. We go on to characterize how the training set size factors into the privacy risks exposed by training a robust model on a simple Gaussian data task, and show empirically that our findings hold on image classification benchmark datasets, such as CIFAR-10 and CIFAR-100.

摘要: 从历史上看，机器学习方法在设计时并没有考虑到安全性。反过来，这又引起了敌意的例子，这些输入样本被仔细扰动，目的是在测试时误导检测，这些样本已被应用于攻击垃圾邮件和恶意软件分类，最近还被应用于攻击图像分类。因此，大量的研究致力于设计对对抗性例子具有鲁棒性的机器学习方法。不幸的是，除了稳健性之外，一个安全的机器学习模型还必须满足公平性和隐私性等要求。宋等人最近的工作。(2019年)的经验表明，稳健的机器学习模型和私人机器学习模型之间存在权衡。与标准(非稳健)模型相比，被设计成对对抗性示例具有鲁棒性的模型通常在更大程度上过度适合于训练数据。如果数据集包含私有信息，那么通过观察模型的输出来分隔训练和测试数据的任何统计测试都可能代表隐私泄露，如果模型过度适合训练数据，这些统计测试就会变得更容易。在这项工作中，我们确定了与稳健模型相比，标准模型将在更大程度上过度拟合的设置，以及在以前的工作中经验观察到的发生相反行为的设置。因此，实现健壮性并不一定要牺牲隐私。过度拟合的程度自然取决于可用于训练的数据量。我们继续描述了通过在一个简单的高斯数据任务上训练一个健壮的模型，训练集大小是如何影响隐私风险的，并实证表明我们的发现适用于图像分类基准数据集，如CIFAR-10和CIFAR-100。



## **30. Detecting CAN Masquerade Attacks with Signal Clustering Similarity**

利用信号聚类相似度检测CAN伪装攻击 cs.CR

7 pages, 7 figures, 1 table

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2201.02665v1)

**Authors**: Pablo Moriano, Robert A. Bridges, Michael D. Iannacone

**Abstracts**: Vehicular Controller Area Networks (CANs) are susceptible to cyber attacks of different levels of sophistication. Fabrication attacks are the easiest to administer -- an adversary simply sends (extra) frames on a CAN -- but also the easiest to detect because they disrupt frame frequency. To overcome time-based detection methods, adversaries must administer masquerade attacks by sending frames in lieu of (and therefore at the expected time of) benign frames but with malicious payloads. Research efforts have proven that CAN attacks, and masquerade attacks in particular, can affect vehicle functionality. Examples include causing unintended acceleration, deactivation of vehicle's brakes, as well as steering the vehicle. We hypothesize that masquerade attacks modify the nuanced correlations of CAN signal time series and how they cluster together. Therefore, changes in cluster assignments should indicate anomalous behavior. We confirm this hypothesis by leveraging our previously developed capability for reverse engineering CAN signals (i.e., CAN-D [Controller Area Network Decoder]) and focus on advancing the state of the art for detecting masquerade attacks by analyzing time series extracted from raw CAN frames. Specifically, we demonstrate that masquerade attacks can be detected by computing time series clustering similarity using hierarchical clustering on the vehicle's CAN signals (time series) and comparing the clustering similarity across CAN captures with and without attacks. We test our approach in a previously collected CAN dataset with masquerade attacks (i.e., the ROAD dataset) and develop a forensic tool as a proof of concept to demonstrate the potential of the proposed approach for detecting CAN masquerade attacks.

摘要: 车辆控制器局域网(CAN)容易受到不同复杂程度的网络攻击。伪造攻击是最容易管理的--对手只是在CAN上发送(额外的)帧--但也是最容易检测到的，因为它们扰乱了帧频率。要克服基于时间的检测方法，攻击者必须通过发送帧来管理伪装攻击，而不是(因此在预期时间发送)良性帧，但带有恶意有效负载。研究工作已经证明，CAN攻击，特别是伪装攻击，会影响车辆的功能。例如，造成意外加速，车辆刹车失灵，以及驾驶车辆。我们假设伪装攻击修改了CAN信号时间序列的细微差别相关性，以及它们是如何聚集在一起的。因此，群集分配中的更改应该表示异常行为。我们利用我们之前开发的CAN信号逆向工程能力(即CAN-D[Controller Area Network Decoder])确认了这一假设，并通过分析从原始CAN帧中提取的时间序列，专注于提高检测伪装攻击的技术水平。具体地说，我们通过对车辆CAN信号(时间序列)进行层次聚类来计算时间序列聚类相似度，并比较有攻击和无攻击的CAN捕获的聚类相似度，从而实现伪装攻击的检测。我们在以前收集的带有伪装攻击的CAN数据集(即道路数据集)上测试了我们的方法，并开发了一个取证工具作为概念证明，以展示所提出的方法检测CAN伪装攻击的潜力。



## **31. Exploring Adversarial Robustness of Multi-Sensor Perception Systems in Self Driving**

自动驾驶中多传感器感知系统的对抗鲁棒性研究 cs.CV

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2101.06784v3)

**Authors**: James Tu, Huichen Li, Xinchen Yan, Mengye Ren, Yun Chen, Ming Liang, Eilyan Bitar, Ersin Yumer, Raquel Urtasun

**Abstracts**: Modern self-driving perception systems have been shown to improve upon processing complementary inputs such as LiDAR with images. In isolation, 2D images have been found to be extremely vulnerable to adversarial attacks. Yet, there have been limited studies on the adversarial robustness of multi-modal models that fuse LiDAR features with image features. Furthermore, existing works do not consider physically realizable perturbations that are consistent across the input modalities. In this paper, we showcase practical susceptibilities of multi-sensor detection by placing an adversarial object on top of a host vehicle. We focus on physically realizable and input-agnostic attacks as they are feasible to execute in practice, and show that a single universal adversary can hide different host vehicles from state-of-the-art multi-modal detectors. Our experiments demonstrate that successful attacks are primarily caused by easily corrupted image features. Furthermore, we find that in modern sensor fusion methods which project image features into 3D, adversarial attacks can exploit the projection process to generate false positives across distant regions in 3D. Towards more robust multi-modal perception systems, we show that adversarial training with feature denoising can boost robustness to such attacks significantly. However, we find that standard adversarial defenses still struggle to prevent false positives which are also caused by inaccurate associations between 3D LiDAR points and 2D pixels.

摘要: 现代自动驾驶感知系统已被证明在处理互补输入(如带有图像的LiDAR)时有所改进。孤立地，2D图像被发现极易受到敌意攻击。然而，关于融合LiDAR特征和图像特征的多模态模型的对抗鲁棒性研究有限。此外，现有的工作没有考虑跨输入模态一致的物理上可实现的扰动。在这篇文章中，我们展示了多传感器检测的实际敏感性，通过在宿主车辆上放置一个敌对对象来实现。我们将重点放在物理上可实现的和输入不可知的攻击上，因为它们在实践中是可行的，并展示了单个通用对手可以隐藏不同的主机车辆，而不是最先进的多模式检测器。我们的实验表明，成功的攻击主要是由容易损坏的图像特征引起的。此外，我们还发现，在将图像特征投影到3D的现代传感器融合方法中，敌意攻击可以利用投影过程在3D中产生跨越遥远区域的假阳性。对于更健壮的多模态感知系统，我们证明了特征去噪的对抗性训练可以显著提高对此类攻击的鲁棒性。然而，我们发现标准的对抗性防御仍然难以防止误报，这也是由于3D LiDAR点和2D像素之间的不准确关联造成的。



## **32. Adversarial Example Detection for DNN Models: A Review and Experimental Comparison**

DNN模型的对抗性范例检测：综述与实验比较 cs.CV

Accepted and published in Artificial Intelligence Review journal

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2105.00203v4)

**Authors**: Ahmed Aldahdooh, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Deforges

**Abstracts**: Deep learning (DL) has shown great success in many human-related tasks, which has led to its adoption in many computer vision based applications, such as security surveillance systems, autonomous vehicles and healthcare. Such safety-critical applications have to draw their path to success deployment once they have the capability to overcome safety-critical challenges. Among these challenges are the defense against or/and the detection of the adversarial examples (AEs). Adversaries can carefully craft small, often imperceptible, noise called perturbations to be added to the clean image to generate the AE. The aim of AE is to fool the DL model which makes it a potential risk for DL applications. Many test-time evasion attacks and countermeasures,i.e., defense or detection methods, are proposed in the literature. Moreover, few reviews and surveys were published and theoretically showed the taxonomy of the threats and the countermeasure methods with little focus in AE detection methods. In this paper, we focus on image classification task and attempt to provide a survey for detection methods of test-time evasion attacks on neural network classifiers. A detailed discussion for such methods is provided with experimental results for eight state-of-the-art detectors under different scenarios on four datasets. We also provide potential challenges and future perspectives for this research direction.

摘要: 深度学习(DL)在许多与人类相关的任务中取得了巨大的成功，这使得它被许多基于计算机视觉的应用所采用，如安全监控系统、自动驾驶汽车和医疗保健。此类安全关键型应用程序一旦具备了克服安全关键型挑战的能力，就必须为成功部署画上句号。在这些挑战中，包括防御或/和检测对抗性示例(AEs)。攻击者可以小心翼翼地制造称为扰动的小噪音，通常是难以察觉的，并将其添加到干净的图像中，以生成AE。AE的目的是愚弄DL模型，使其成为DL应用程序的潜在风险。文献中提出了许多测试时间逃避攻击和对策，即防御或检测方法。此外，很少有综述和调查发表，从理论上给出了威胁的分类和对策方法，而对声发射检测方法的关注较少。本文以图像分类任务为研究对象，对神经网络分类器测试时间逃避攻击的检测方法进行了综述。对这些方法进行了详细的讨论，并给出了在四个数据集上的不同场景下八个最先进检测器的实验结果。我们还对这一研究方向提出了潜在的挑战和未来的展望。



## **33. Semantically Stealthy Adversarial Attacks against Segmentation Models**

针对分割模型的语义隐蔽敌意攻击 cs.CV

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2104.01732v3)

**Authors**: Zhenhua Chen, Chuhua Wang, David J. Crandall

**Abstracts**: Segmentation models have been found to be vulnerable to targeted and non-targeted adversarial attacks. However, the resulting segmentation outputs are often so damaged that it is easy to spot an attack. In this paper, we propose semantically stealthy adversarial attacks which can manipulate targeted labels while preserving non-targeted labels at the same time. One challenge is making semantically meaningful manipulations across datasets and models. Another challenge is avoiding damaging non-targeted labels. To solve these challenges, we consider each input image as prior knowledge to generate perturbations. We also design a special regularizer to help extract features. To evaluate our model's performance, we design three basic attack types, namely `vanishing into the context,' `embedding fake labels,' and `displacing target objects.' Our experiments show that our stealthy adversarial model can attack segmentation models with a relatively high success rate on Cityscapes, Mapillary, and BDD100K. Our framework shows good empirical generalization across datasets and models.

摘要: 已经发现，分段模型容易受到目标攻击和非目标攻击。然而，由此产生的分段输出通常损坏得非常严重，因此很容易发现攻击。在本文中，我们提出了一种语义隐蔽的敌意攻击，它可以在操作目标标签的同时保留非目标标签。一个挑战是跨数据集和模型进行语义上有意义的操作。另一个挑战是避免损坏非目标标签。为了解决这些挑战，我们将每一幅输入图像作为先验知识来产生扰动。我们还设计了一个特殊的正则化器来帮助提取特征。为了评估模型的性能，我们设计了三种基本的攻击类型，即“消失在上下文中”、“嵌入假标签”和“移位目标对象”。我们的实验表明，我们的隐形攻击模型能够在城市景观、Mapillary和BDD100K上以相对较高的成功率攻击分割模型。我们的框架在数据集和模型之间显示了良好的经验概括性。



## **34. Towards Understanding and Harnessing the Effect of Image Transformation in Adversarial Detection**

认识和利用图像变换在对抗性检测中的作用 cs.CV

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2201.01080v2)

**Authors**: Hui Liu, Bo Zhao, Yuefeng Peng, Weidong Li, Peng Liu

**Abstracts**: Deep neural networks (DNNs) are threatened by adversarial examples. Adversarial detection, which distinguishes adversarial images from benign images, is fundamental for robust DNN-based services. Image transformation is one of the most effective approaches to detect adversarial examples. During the last few years, a variety of image transformations have been studied and discussed to design reliable adversarial detectors. In this paper, we systematically synthesize the recent progress on adversarial detection via image transformations with a novel classification method. Then, we conduct extensive experiments to test the detection performance of image transformations against state-of-the-art adversarial attacks. Furthermore, we reveal that each individual transformation is not capable of detecting adversarial examples in a robust way, and propose a DNN-based approach referred to as AdvJudge, which combines scores of 9 image transformations. Without knowing which individual scores are misleading or not misleading, AdvJudge can make the right judgment, and achieve a significant improvement in detection accuracy. We claim that AdvJudge is a more effective adversarial detector than those based on an individual image transformation.

摘要: 深度神经网络(DNNs)受到敌意例子的威胁。敌意检测将敌意图像与良性图像区分开来，是健壮的基于DNN的服务的基础。图像变换是检测敌意例子最有效的方法之一。在过去的几年里，人们对各种图像变换进行了研究和讨论，以设计可靠的对抗性检测器。本文采用一种新的分类方法，系统地综述了基于图像变换的敌意检测的最新进展。然后，我们进行了大量的实验来测试图像变换对最新的敌意攻击的检测性能。此外，我们还揭示了每个个体变换并不能稳健地检测敌意示例，并提出了一种基于DNN的方法，称为AdvJustice，它结合了9个图像变换的分数。在不知道哪些分数具有误导性或不具有误导性的情况下，AdvJustice可以做出正确的判断，并实现检测准确率的显著提高。我们认为，与基于个体图像变换的检测器相比，AdvJustice是一种更有效的对抗性检测器。



## **35. BDFA: A Blind Data Adversarial Bit-flip Attack on Deep Neural Networks**

BDFA：一种基于深度神经网络的盲数据对抗性比特翻转攻击 cs.CR

**SubmitDate**: 2022-01-07    [paper-pdf](http://arxiv.org/pdf/2112.03477v2)

**Authors**: Behnam Ghavami, Mani Sadati, Mohammad Shahidzadeh, Zhenman Fang, Lesley Shannon

**Abstracts**: Adversarial bit-flip attack (BFA) on Neural Network weights can result in catastrophic accuracy degradation by flipping a very small number of bits. A major drawback of prior bit flip attack techniques is their reliance on test data. This is frequently not possible for applications that contain sensitive or proprietary data. In this paper, we propose Blind Data Adversarial Bit-flip Attack (BDFA), a novel technique to enable BFA without any access to the training or testing data. This is achieved by optimizing for a synthetic dataset, which is engineered to match the statistics of batch normalization across different layers of the network and the targeted label. Experimental results show that BDFA could decrease the accuracy of ResNet50 significantly from 75.96\% to 13.94\% with only 4 bits flips.

摘要: 对神经网络权重的对抗性比特翻转攻击(BFA)可以通过翻转非常少量的比特来导致灾难性的精度降低。现有比特翻转攻击技术的主要缺点是它们对测试数据的依赖。对于包含敏感或专有数据的应用程序而言，这通常是不可能的。在本文中，我们提出了盲数据对抗比特翻转攻击(BDFA)，这是一种新的技术，可以在不访问任何训练或测试数据的情况下实现BFA。这是通过对合成数据集进行优化来实现的，该合成数据集被设计为匹配跨网络的不同层和目标标签的批归一化的统计数据。实验结果表明，只需4位翻转，BDFA就能将ResNet50的精度从75.96降到13.94。



## **36. Accelerated Zeroth-Order and First-Order Momentum Methods from Mini to Minimax Optimization**

从Mini到Minimax优化的加速零阶和一阶动量方法 math.OC

Published in Journal of Machine Learning Research (JMLR)

**SubmitDate**: 2022-01-06    [paper-pdf](http://arxiv.org/pdf/2008.08170v6)

**Authors**: Feihu Huang, Shangqian Gao, Jian Pei, Heng Huang

**Abstracts**: In the paper, we propose a class of accelerated zeroth-order and first-order momentum methods for both nonconvex mini-optimization and minimax-optimization. Specifically, we propose a new accelerated zeroth-order momentum (Acc-ZOM) method for black-box mini-optimization. Moreover, we prove that our Acc-ZOM method achieves a lower query complexity of $\tilde{O}(d^{3/4}\epsilon^{-3})$ for finding an $\epsilon$-stationary point, which improves the best known result by a factor of $O(d^{1/4})$ where $d$ denotes the variable dimension. In particular, the Acc-ZOM does not require large batches required in the existing zeroth-order stochastic algorithms. Meanwhile, we propose an accelerated \textbf{zeroth-order} momentum descent ascent (Acc-ZOMDA) method for \textbf{black-box} minimax-optimization, which obtains a query complexity of $\tilde{O}((d_1+d_2)^{3/4}\kappa_y^{4.5}\epsilon^{-3})$ without large batches for finding an $\epsilon$-stationary point, where $d_1$ and $d_2$ denote variable dimensions and $\kappa_y$ is condition number. Moreover, we propose an accelerated \textbf{first-order} momentum descent ascent (Acc-MDA) method for \textbf{white-box} minimax optimization, which has a gradient complexity of $\tilde{O}(\kappa_y^{4.5}\epsilon^{-3})$ without large batches for finding an $\epsilon$-stationary point. In particular, our Acc-MDA can obtain a lower gradient complexity of $\tilde{O}(\kappa_y^{2.5}\epsilon^{-3})$ with a batch size $O(\kappa_y^4)$. Extensive experimental results on black-box adversarial attack to deep neural networks and poisoning attack to logistic regression demonstrate efficiency of our algorithms.

摘要: 本文提出了一类求解非凸极小优化和极小极大优化的加速零阶和一阶动量方法。具体地说，我们提出了一种新的加速零阶动量(ACC-ZOM)方法来求解黑盒最小优化问题。此外，我们还证明了我们的ACC-zom方法在寻找$\epsilon$-驻点时的查询复杂度为$tilde{O}(d^{3/4}\epsilon^{-3})$，这将最著名的结果改进了$O(d^{1/4})$，其中$d$表示变量维数。特别地，ACC-ZOM不需要现有零阶随机算法所要求的大批量。同时，我们提出了一种求解Textbf{黑盒}极小极大优化问题的加速动量下降上升方法，该方法无需大批量即可获得$\tilde{O}((d_1+d_2)^{3/4}\kappa_y^{4.5}\epsilon^{-3})$的查询复杂度，其中$d1$和$d2$表示可变维数，$kappay$表示条件数。此外，我们还提出了一种加速的textbf{一阶}动量下降上升(ACC-MDA)方法来求解textbf{白盒}极小极大优化问题，该方法的梯度复杂度为$tide{O}(\kappa_y^{4.5}\epsilon^{-3})$，不需要大批量地寻找$\epsilon$-驻点。特别地，在批量为$O(\kappa_y^4)$的情况下，我们的ACC-MDA可以获得$\tide{O}(\kappa_y^{2.5}\epsilon^{-3})$的较低梯度复杂度。对深层神经网络的黑盒攻击和对Logistic回归的中毒攻击的大量实验结果表明了该算法的有效性。



## **37. Increased-confidence adversarial examples for deep learning counter-forensics**

深度学习反取证的增信度对抗性实例 cs.CV

**SubmitDate**: 2022-01-06    [paper-pdf](http://arxiv.org/pdf/2005.06023v2)

**Authors**: Wenjie Li, Benedetta Tondi, Rongrong Ni, Mauro Barni

**Abstracts**: Transferability of adversarial examples is a key issue to apply this kind of attacks against multimedia forensics (MMF) techniques based on Deep Learning (DL) in a real-life setting. Adversarial example transferability, in fact, would open the way to the deployment of successful counter forensics attacks also in cases where the attacker does not have a full knowledge of the to-be-attacked system. Some preliminary works have shown that adversarial examples against CNN-based image forensics detectors are in general non-transferrable, at least when the basic versions of the attacks implemented in the most popular libraries are adopted. In this paper, we introduce a general strategy to increase the strength of the attacks and evaluate their transferability when such a strength varies. We experimentally show that, in this way, attack transferability can be largely increased, at the expense of a larger distortion. Our research confirms the security threats posed by the existence of adversarial examples even in multimedia forensics scenarios, thus calling for new defense strategies to improve the security of DL-based MMF techniques.

摘要: 对抗性例子的可移植性是将这类基于深度学习的多媒体取证攻击技术应用于实际环境的关键问题。事实上，对抗性示例的可转移性将为部署成功的反取证攻击开辟道路，在攻击者不完全了解要攻击的系统的情况下也是如此。一些初步工作表明，针对基于CNN的图像取证检测器的敌意示例通常是不可转移的，至少当采用在最流行的库中实现的攻击的基本版本时是如此。在本文中，我们介绍了一种增加攻击强度的一般策略，并在攻击强度变化时评估其可转移性。我们的实验表明，通过这种方式，攻击的可传递性可以大大提高，但代价是更大的失真。我们的研究证实了即使在多媒体取证场景中，敌意例子的存在也会带来安全威胁，因此需要新的防御策略来提高基于DL的MMF技术的安全性。



## **38. HydraText: Multi-objective Optimization for Adversarial Textual Attack**

HydraText：对抗性文本攻击的多目标优化 cs.CL

**SubmitDate**: 2022-01-06    [paper-pdf](http://arxiv.org/pdf/2111.01528v2)

**Authors**: Shengcai Liu, Ning Lu, Wenjing Hong, Chao Qian, Ke Tang

**Abstracts**: The field of adversarial textual attack has significantly grown over the last few years, where the commonly considered objective is to craft adversarial examples (AEs) that can successfully fool the target model. However, the imperceptibility of attacks, which is also an essential objective for practical attackers, is often left out by previous studies. In consequence, the crafted AEs tend to have obvious structural and semantic differences from the original human-written texts, making them easily perceptible. In this paper, we advocate simultaneously considering both objectives of successful and imperceptible attacks. Specifically, we formulate the problem of crafting AEs as a multi-objective set maximization problem, and propose a novel evolutionary algorithm (dubbed HydraText) to solve it. To the best of our knowledge, HydraText is currently the only approach that can be effectively applied to both score-based and decision-based attack settings. Exhaustive experiments involving 44237 instances demonstrate that HydraText consistently achieves higher attack success rates and better attack imperceptibility than the state-of-the-art textual attack approaches. A human evaluation study also shows that the AEs crafted by HydraText are more indistinguishable from human-written texts. Finally, these AEs exhibit good transferability and can bring notable robustness improvement to the target models by adversarial training.

摘要: 对抗性文本攻击领域在过去几年中显著增长，其中通常被认为的目标是制作能够成功愚弄目标模型的对抗性示例(AEs)。然而，攻击的隐蔽性也是实际攻击者的一个重要目标，但以往的研究往往忽略了这一点。因此，精心制作的AE往往与原始的人类书写的文本在结构和语义上有明显的差异，使得它们很容易被察觉。在本文中，我们主张同时考虑成功攻击和隐蔽攻击的目标。具体地说，我们将AES的制作问题描述为一个多目标集合最大化问题，并提出了一种新的进化算法(称为HydraText)来求解该问题。据我们所知，HydraText是目前唯一可以有效应用于基于分数和基于决策的攻击设置的方法。涉及44237个实例的详尽实验表明，与最先进的文本攻击方法相比，HydraText始终具有更高的攻击成功率和更好的攻击隐蔽性。一项人类评估研究也表明，HydraText制作的AE与人类书写的文本更难以区分。最后，这些AEs表现出良好的可移植性，通过对抗性训练可以显著提高目标模型的鲁棒性。



## **39. Increasing the Confidence of Deep Neural Networks by Coverage Analysis**

利用覆盖分析提高深度神经网络的可信度 cs.LG

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2101.12100v3)

**Authors**: Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: The great performance of machine learning algorithms and deep neural networks in several perception and control tasks is pushing the industry to adopt such technologies in safety-critical applications, as autonomous robots and self-driving vehicles. At present, however, several issues need to be solved to make deep learning methods more trustworthy, predictable, safe, and secure against adversarial attacks. Although several methods have been proposed to improve the trustworthiness of deep neural networks, most of them are tailored for specific classes of adversarial examples, hence failing to detect other corner cases or unsafe inputs that heavily deviate from the training samples. This paper presents a lightweight monitoring architecture based on coverage paradigms to enhance the model robustness against different unsafe inputs. In particular, four coverage analysis methods are proposed and tested in the architecture for evaluating multiple detection logics. Experimental results show that the proposed approach is effective in detecting both powerful adversarial examples and out-of-distribution inputs, introducing limited extra-execution time and memory requirements.

摘要: 机器学习算法和深度神经网络在几个感知和控制任务中的出色性能，正在推动该行业在自动机器人和自动驾驶汽车等安全关键应用中采用这些技术。然而，目前还需要解决几个问题，以使深度学习方法更可信、更可预测、更安全、更能抵御对手攻击。虽然已经提出了几种方法来提高深度神经网络的可信性，但大多数方法都是针对特定类别的对抗性示例而定制的，因此无法检测出严重偏离训练样本的其他角点情况或不安全输入。提出了一种基于复盖范型的轻量级监控体系结构，以增强模型对不同不安全输入的鲁棒性。特别地，提出了四种覆盖分析方法，并在该体系结构中进行了测试，以评估多个检测逻辑。实验结果表明，该方法能有效地检测出强大的对抗性实例和分布不均的输入，并引入有限的额外执行时间和内存需求。



## **40. On the Real-World Adversarial Robustness of Real-Time Semantic Segmentation Models for Autonomous Driving**

自动驾驶实时语义分割模型的真实对抗性研究 cs.CV

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2201.01850v1)

**Authors**: Giulio Rossolini, Federico Nesti, Gianluca D'Amico, Saasha Nair, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: The existence of real-world adversarial examples (commonly in the form of patches) poses a serious threat for the use of deep learning models in safety-critical computer vision tasks such as visual perception in autonomous driving. This paper presents an extensive evaluation of the robustness of semantic segmentation models when attacked with different types of adversarial patches, including digital, simulated, and physical ones. A novel loss function is proposed to improve the capabilities of attackers in inducing a misclassification of pixels. Also, a novel attack strategy is presented to improve the Expectation Over Transformation method for placing a patch in the scene. Finally, a state-of-the-art method for detecting adversarial patch is first extended to cope with semantic segmentation models, then improved to obtain real-time performance, and eventually evaluated in real-world scenarios. Experimental results reveal that, even though the adversarial effect is visible with both digital and real-world attacks, its impact is often spatially confined to areas of the image around the patch. This opens to further questions about the spatial robustness of real-time semantic segmentation models.

摘要: 真实世界中对抗性例子(通常以补丁的形式)的存在严重威胁了深度学习模型在安全关键的计算机视觉任务中的应用，例如自动驾驶中的视觉感知。本文对语义分割模型在受到不同类型的敌意补丁(包括数字、模拟和物理补丁)攻击时的健壮性进行了广泛的评估。提出了一种新的损失函数，以提高攻击者诱导像素误分类的能力。同时，提出了一种新的攻击策略，改进了在场景中放置补丁的期望超变换法。最后，对现有的恶意补丁检测方法进行了扩展，使之适用于语义分割模型，并对其进行了改进以获得实时性能，最后在真实场景中进行了评估。实验结果表明，即使在数字和真实世界的攻击中都可以看到对抗效果，但其影响通常局限于补丁周围的图像区域。这对实时语义分割模型的空间稳健性提出了进一步的问题。



## **41. Secure Remote Attestation with Strong Key Insulation Guarantees**

具有强密钥绝缘性保证的安全远程证明 cs.CR

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2201.01834v1)

**Authors**: Deniz Gurevin, Chenglu Jin, Phuong Ha Nguyen, Omer Khan, Marten van Dijk

**Abstracts**: Recent years have witnessed a trend of secure processor design in both academia and industry. Secure processors with hardware-enforced isolation can be a solid foundation of cloud computation in the future. However, due to recent side-channel attacks, the commercial secure processors failed to deliver the promises of a secure isolated execution environment. Sensitive information inside the secure execution environment always gets leaked via side channels. This work considers the most powerful software-based side-channel attackers, i.e., an All Digital State Observing (ADSO) adversary who can observe all digital states, including all digital states in secure enclaves. Traditional signature schemes are not secure in ADSO adversarial model. We introduce a new cryptographic primitive called One-Time Signature with Secret Key Exposure (OTS-SKE), which ensures no one can forge a valid signature of a new message or nonce even if all secret session keys are leaked. OTS-SKE enables us to sign attestation reports securely under the ADSO adversary. We also minimize the trusted computing base by introducing a secure co-processor into the system, and the interaction between the secure co-processor and the attestation processor is unidirectional. That is, the co-processor takes no inputs from the processor and only generates secret keys for the processor to fetch. Our experimental results show that the signing of OTS-SKE is faster than that of Elliptic Curve Digital Signature Algorithm (ECDSA) used in Intel SGX.

摘要: 近年来，学术界和工业界都出现了安全处理器设计的趋势。具有硬件强制隔离的安全处理器可以成为未来云计算的坚实基础。然而，由于最近的侧信道攻击，商业安全处理器未能实现安全隔离执行环境的承诺。安全执行环境中的敏感信息总是通过旁路泄露。这项工作考虑了最强大的基于软件的侧信道攻击者，即可以观察所有数字状态(包括安全飞地中的所有数字状态)的全数字状态观测(ADSO)对手。传统的签名方案在Adso对抗模型下是不安全的。我们引入了一种新的密码原语，称为具有密钥暴露的一次性签名(OTS-SKE)，它确保即使所有的秘密会话密钥都被泄露，也没有人能够伪造新消息或随机数的有效签名。OTS-SKE使我们能够在Adso对手下安全地签署证明报告。我们还通过在系统中引入安全协处理器来最小化可信计算基，并且安全协处理器和证明处理器之间的交互是单向的。也就是说，协处理器不接受来自处理器的输入，而只生成密钥供处理器提取。实验结果表明，OTS-SKE的签名速度比Intel SGX中使用的椭圆曲线数字签名算法(ECDSA)要快。



## **42. Generation of Wheel Lockup Attacks on Nonlinear Dynamics of Vehicle Traction**

车辆牵引非线性动力学中车轮闭锁攻击的产生 eess.SY

Submitted to AutoSec'22@NDSS

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2112.09229v2)

**Authors**: Alireza Mohammadi, Hafiz Malik, Masoud Abbaszadeh

**Abstracts**: There is ample evidence in the automotive cybersecurity literature that the car brake ECUs can be maliciously reprogrammed. Motivated by such threat, this paper investigates the capabilities of an adversary who can directly control the frictional brake actuators and would like to induce wheel lockup conditions leading to catastrophic road injuries. This paper demonstrates that the adversary despite having a limited knowledge of the tire-road interaction characteristics has the capability of driving the states of the vehicle traction dynamics to a vicinity of the lockup manifold in a finite time by means of a properly designed attack policy for the frictional brakes. This attack policy relies on employing a predefined-time controller and a nonlinear disturbance observer acting on the wheel slip error dynamics. Simulations under various road conditions demonstrate the effectiveness of the proposed attack policy.

摘要: 汽车网络安全文献中有大量证据表明，汽车刹车ECU可以被恶意重新编程。在这种威胁的驱使下，本文调查了一个可以直接控制摩擦制动执行器并想要诱导车轮锁定条件导致灾难性道路伤害的对手的能力。通过合理设计摩擦制动器的攻击策略，证明了敌方尽管对轮胎-路面相互作用特性知之甚少，但仍有能力在有限的时间内将车辆牵引动力学状态驱动到闭锁歧管附近。该攻击策略依赖于采用预定义时间控制器和作用于车轮打滑误差动态的非线性扰动观测器。在不同路况下的仿真实验验证了所提出的攻击策略的有效性。



## **43. GRNN: Generative Regression Neural Network -- A Data Leakage Attack for Federated Learning**

GRNN：产生式回归神经网络--一种面向联邦学习的数据泄漏攻击 cs.LG

The source code can be found at: https://github.com/Rand2AI/GRNN

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2105.00529v2)

**Authors**: Hanchi Ren, Jingjing Deng, Xianghua Xie

**Abstracts**: Data privacy has become an increasingly important issue in Machine Learning (ML), where many approaches have been developed to tackle this challenge, e.g. cryptography (Homomorphic Encryption (HE), Differential Privacy (DP), etc.) and collaborative training (Secure Multi-Party Computation (MPC), Distributed Learning and Federated Learning (FL)). These techniques have a particular focus on data encryption or secure local computation. They transfer the intermediate information to the third party to compute the final result. Gradient exchanging is commonly considered to be a secure way of training a robust model collaboratively in Deep Learning (DL). However, recent researches have demonstrated that sensitive information can be recovered from the shared gradient. Generative Adversarial Network (GAN), in particular, has shown to be effective in recovering such information. However, GAN based techniques require additional information, such as class labels which are generally unavailable for privacy-preserved learning. In this paper, we show that, in the FL system, image-based privacy data can be easily recovered in full from the shared gradient only via our proposed Generative Regression Neural Network (GRNN). We formulate the attack to be a regression problem and optimize two branches of the generative model by minimizing the distance between gradients. We evaluate our method on several image classification tasks. The results illustrate that our proposed GRNN outperforms state-of-the-art methods with better stability, stronger robustness, and higher accuracy. It also has no convergence requirement to the global FL model. Moreover, we demonstrate information leakage using face re-identification. Some defense strategies are also discussed in this work.

摘要: 数据隐私已经成为机器学习(ML)中一个越来越重要的问题，已经有许多方法被开发出来来应对这一挑战，例如密码学(同态加密(HE)、差分隐私(DP)等)。和协作培训(安全多方计算(MPC)、分布式学习和联合学习(FL))。这些技术特别关注数据加密或安全本地计算。他们将中间信息传递给第三方来计算最终结果。梯度交换通常被认为是深度学习中协作训练鲁棒模型的一种安全方式。然而，最近的研究表明，敏感信息可以从共享梯度中恢复出来。尤其是生成性对抗性网络(GAN)在恢复此类信息方面表现得尤为有效。然而，基于GAN的技术需要额外的信息，例如通常不能用于隐私保护学习的类别标签。在本文中，我们证明了在FL系统中，仅通过我们提出的生成回归神经网络(GRNN)就可以很容易地从共享梯度中完全恢复基于图像的隐私数据。我们将攻击描述为一个回归问题，并通过最小化梯度之间的距离来优化生成模型的两个分支。我们在几个图像分类任务上对我们的方法进行了评估。实验结果表明，本文提出的GRNN方法在稳定性、鲁棒性和精确度等方面均优于目前最先进的方法。它对全局FL模型也没有收敛要求。此外，我们使用人脸重新识别来演示信息泄漏。文中还讨论了一些防御策略。



## **44. Aspis: A Robust Detection System for Distributed Learning**

ASPIS：一种健壮的分布式学习检测系统 cs.LG

17 pages, 23 figures

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2108.02416v2)

**Authors**: Konstantinos Konstantinidis, Aditya Ramamoorthy

**Abstracts**: State-of-the-art machine learning models are routinely trained on large-scale distributed clusters. Crucially, such systems can be compromised when some of the computing devices exhibit abnormal (Byzantine) behavior and return arbitrary results to the parameter server (PS). This behavior may be attributed to a plethora of reasons, including system failures and orchestrated attacks. Existing work suggests robust aggregation and/or computational redundancy to alleviate the effect of distorted gradients. However, most of these schemes are ineffective when an adversary knows the task assignment and can choose the attacked workers judiciously to induce maximal damage. Our proposed method Aspis assigns gradient computations to worker nodes using a subset-based assignment which allows for multiple consistency checks on the behavior of a worker node. Examination of the calculated gradients and post-processing (clique-finding in an appropriately constructed graph) by the central node allows for efficient detection and subsequent exclusion of adversaries from the training process. We prove the Byzantine resilience and detection guarantees of Aspis under weak and strong attacks and extensively evaluate the system on various large-scale training scenarios. The principal metric for our experiments is the test accuracy, for which we demonstrate a significant improvement of about 30% compared to many state-of-the-art approaches on the CIFAR-10 dataset. The corresponding reduction of the fraction of corrupted gradients ranges from 16% to 99%.

摘要: 最先进的机器学习模型通常在大规模分布式集群上进行训练。重要的是，当一些计算设备表现出异常(拜占庭)行为并向参数服务器(PS)返回任意结果时，这样的系统可能受到危害。此行为可能归因于过多的原因，包括系统故障和精心策划的攻击。现有的工作表明，稳健的聚集和/或计算冗余可以减轻扭曲梯度的影响。然而，当敌手知道任务分配，并且能够明智地选择被攻击的工人以造成最大的损害时，这些方案大多是无效的。我们提出的方法ASPIS使用基于子集的分配将梯度计算分配给工作节点，这允许对工作节点的行为进行多个一致性检查。由中心节点检查计算出的梯度和后处理(在适当构造的图中发现集团)允许有效地检测并随后将对手排除在训练过程之外。我们证明了ASPIS在弱攻击和强攻击下的拜占庭抗攻击能力和检测保证，并在各种大规模训练场景下对该系统进行了广泛的评估。我们实验的主要指标是测试精度，在CIFAR-10数据集上，与许多最先进的方法相比，我们证明了这一点有大约30%的显著改进。受污染梯度比例的相应降低范围为16%至99%。



## **45. ROOM: Adversarial Machine Learning Attacks Under Real-Time Constraints**

房间：实时约束下的对抗性机器学习攻击 cs.CR

12 pages

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2201.01621v1)

**Authors**: Amira Guesmi, Khaled N. Khasawneh, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstracts**: Advances in deep learning have enabled a wide range of promising applications. However, these systems are vulnerable to Adversarial Machine Learning (AML) attacks; adversarially crafted perturbations to their inputs could cause them to misclassify. Several state-of-the-art adversarial attacks have demonstrated that they can reliably fool classifiers making these attacks a significant threat. Adversarial attack generation algorithms focus primarily on creating successful examples while controlling the noise magnitude and distribution to make detection more difficult. The underlying assumption of these attacks is that the adversarial noise is generated offline, making their execution time a secondary consideration. However, recently, just-in-time adversarial attacks where an attacker opportunistically generates adversarial examples on the fly have been shown to be possible. This paper introduces a new problem: how do we generate adversarial noise under real-time constraints to support such real-time adversarial attacks? Understanding this problem improves our understanding of the threat these attacks pose to real-time systems and provides security evaluation benchmarks for future defenses. Therefore, we first conduct a run-time analysis of adversarial generation algorithms. Universal attacks produce a general attack offline, with no online overhead, and can be applied to any input; however, their success rate is limited because of their generality. In contrast, online algorithms, which work on a specific input, are computationally expensive, making them inappropriate for operation under time constraints. Thus, we propose ROOM, a novel Real-time Online-Offline attack construction Model where an offline component serves to warm up the online algorithm, making it possible to generate highly successful attacks under time constraints.

摘要: 深度学习的进展使其具有广泛的应用前景。然而，这些系统很容易受到对抗性机器学习(AML)攻击；对它们的输入进行恶意设计的扰动可能会导致它们错误分类。几个最先进的对抗性攻击已经证明，它们可以可靠地欺骗分类器，使这些攻击成为一个重大威胁。对抗性攻击生成算法主要集中在创建成功的示例，同时控制噪声的大小和分布，使检测变得更加困难。这些攻击的基本假设是敌意噪音是离线生成的，这使得它们的执行时间成为次要考虑因素。然而，最近，攻击者在飞翔上机会主义地生成对抗性示例的即时对抗性攻击已被证明是可能的。本文提出了一个新的问题：如何在实时约束下产生对抗性噪声来支持这种实时对抗性攻击？了解这一问题有助于我们理解这些攻击对实时系统构成的威胁，并为未来的防御提供安全评估基准。因此，我们首先对对抗性生成算法进行运行时分析。通用攻击离线产生一般攻击，没有在线开销，可以应用于任何输入；但是，由于其通用性，其成功率是有限的。相比之下，处理特定输入的在线算法计算成本较高，因此不适合在时间限制下运行。因此，我们提出了一种新颖的实时在线-离线攻击构建模型--ROOM，其中离线组件用于在线算法的预热，使得在时间约束下生成高度成功的攻击成为可能。



## **46. A Survey on Adversarial Attacks for Malware Analysis**

面向恶意软件分析的敌意攻击研究综述 cs.CR

48 Pages, 31 Figures, 11 Tables

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2111.08223v2)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam

**Abstracts**: Machine learning has witnessed tremendous growth in its adoption and advancement in the last decade. The evolution of machine learning from traditional algorithms to modern deep learning architectures has shaped the way today's technology functions. Its unprecedented ability to discover knowledge/patterns from unstructured data and automate the decision-making process led to its application in wide domains. High flying machine learning arena has been recently pegged back by the introduction of adversarial attacks. Adversaries are able to modify data, maximizing the classification error of the models. The discovery of blind spots in machine learning models has been exploited by adversarial attackers by generating subtle intentional perturbations in test samples. Increasing dependency on data has paved the blueprint for ever-high incentives to camouflage machine learning models. To cope with probable catastrophic consequences in the future, continuous research is required to find vulnerabilities in form of adversarial and design remedies in systems. This survey aims at providing the encyclopedic introduction to adversarial attacks that are carried out against malware detection systems. The paper will introduce various machine learning techniques used to generate adversarial and explain the structure of target files. The survey will also model the threat posed by the adversary and followed by brief descriptions of widely accepted adversarial algorithms. Work will provide a taxonomy of adversarial evasion attacks on the basis of attack domain and adversarial generation techniques. Adversarial evasion attacks carried out against malware detectors will be discussed briefly under each taxonomical headings and compared with concomitant researches. Analyzing the current research challenges in an adversarial generation, the survey will conclude by pinpointing the open future research directions.

摘要: 在过去的十年里，机器学习在其采用和发展方面取得了巨大的增长。机器学习从传统算法到现代深度学习体系结构的演变塑造了当今技术的运作方式。它前所未有的从非结构化数据中发现知识/模式并使决策过程自动化的能力使其在广泛的领域得到了应用。最近，由于对抗性攻击的引入，高飞行机器学习领域受到了阻碍。攻击者能够修改数据，最大化模型的分类错误。机器学习模型中盲点的发现已经被敌意攻击者通过在测试样本中产生微妙的故意扰动来利用。对数据的日益依赖已经为伪装机器学习模型的持续高额激励铺平了蓝图。为了应对未来可能出现的灾难性后果，需要不断进行研究，以发现系统中对抗性形式的漏洞和设计补救措施。本调查旨在提供针对恶意软件检测系统进行的敌意攻击的百科全书介绍。本文将介绍用于生成对抗性的各种机器学习技术，并解释目标文件的结构。调查还将模拟对手构成的威胁，随后简要描述被广泛接受的对抗性算法。工作将提供基于攻击域和敌意生成技术的对抗性逃避攻击的分类。针对恶意软件检测器进行的敌意规避攻击将在每个分类标题下简要讨论，并与相应的研究进行比较。通过分析当前对抗性世代的研究挑战，本调查将通过指出开放的未来研究方向来得出结论。



## **47. Fast Gradient Non-sign Methods**

快速梯度无符号方法 cs.CV

**SubmitDate**: 2022-01-05    [paper-pdf](http://arxiv.org/pdf/2110.12734v2)

**Authors**: Yaya Cheng, Xiaosu Zhu, Qilong Zhang, Lianli Gao, Jingkuan Song

**Abstracts**: Adversarial attacks make their success in "fooling" DNNs and among them, gradient-based algorithms become one of the mainstreams. Based on the linearity hypothesis [12], under $\ell_\infty$ constraint, $sign$ operation applied to the gradients is a good choice for generating perturbations. However, the side-effect from such operation exists since it leads to the bias of direction between the real gradients and the perturbations. In other words, current methods contain a gap between real gradients and actual noises, which leads to biased and inefficient attacks. Therefore in this paper, based on the Taylor expansion, the bias is analyzed theoretically and the correction of $\sign$, i.e., Fast Gradient Non-sign Method (FGNM), is further proposed. Notably, FGNM is a general routine, which can seamlessly replace the conventional $sign$ operation in gradient-based attacks with negligible extra computational cost. Extensive experiments demonstrate the effectiveness of our methods. Specifically, ours outperform them by \textbf{27.5\%} at most and \textbf{9.5\%} on average. Our anonymous code is publicly available: \url{https://git.io/mm-fgnm}.

摘要: 敌意攻击使它们成功地“欺骗”了DNN，其中基于梯度的算法成为主流之一。基于线性假设[12]，在$\ell_\infty$约束下，对梯度进行$sign$运算是产生扰动的一个很好的选择。然而，这种操作的副作用是存在的，因为它导致了真实梯度和扰动之间的方向偏差。换言之，目前的方法存在真实梯度和实际噪声之间的差距，这导致了有偏的、低效的攻击。因此，本文在泰勒展开的基础上，对偏差进行了理论分析，并进一步提出了符号修正方法，即快速梯度无符号方法(FGNM)。值得注意的是，FGNM是一个通用例程，在基于梯度的攻击中可以无缝地取代传统的$sign$操作，而额外的计算代价可以忽略不计。大量实验证明了该方法的有效性。具体地说，我们的性能最多比它们高\textbf{27.5\%}，平均\textbf{9.5\%}。我们的匿名代码是公开提供的：\url{https://git.io/mm-fgnm}.



## **48. Adversarial Feature Desensitization**

对抗性特征脱敏 cs.LG

Accepted at Neurips 2021

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2006.04621v3)

**Authors**: Pouya Bashivan, Reza Bayat, Adam Ibrahim, Kartik Ahuja, Mojtaba Faramarzi, Touraj Laleh, Blake Aaron Richards, Irina Rish

**Abstracts**: Neural networks are known to be vulnerable to adversarial attacks -- slight but carefully constructed perturbations of the inputs which can drastically impair the network's performance. Many defense methods have been proposed for improving robustness of deep networks by training them on adversarially perturbed inputs. However, these models often remain vulnerable to new types of attacks not seen during training, and even to slightly stronger versions of previously seen attacks. In this work, we propose a novel approach to adversarial robustness, which builds upon the insights from the domain adaptation field. Our method, called Adversarial Feature Desensitization (AFD), aims at learning features that are invariant towards adversarial perturbations of the inputs. This is achieved through a game where we learn features that are both predictive and robust (insensitive to adversarial attacks), i.e. cannot be used to discriminate between natural and adversarial data. Empirical results on several benchmarks demonstrate the effectiveness of the proposed approach against a wide range of attack types and attack strengths. Our code is available at https://github.com/BashivanLab/afd.

摘要: 众所周知，神经网络容易受到对抗性攻击--轻微但精心构建的输入扰动会严重损害网络的性能。为了提高深层网络的鲁棒性，已经提出了许多防御方法，通过对深层网络进行恶意扰动输入的训练来提高它们的健壮性。然而，这些模型经常仍然容易受到训练期间未见的新类型攻击的攻击，甚至对以前见过的略强版本的攻击也是如此。在这项工作中，我们提出了一种新的方法来实现对手鲁棒性，该方法建立在领域自适应领域的洞察力的基础上。我们的方法被称为对抗性特征脱敏(AFD)，目的是学习对输入的对抗性扰动不变的特征。这是通过一个游戏来实现的，在这个游戏中，我们学习了既具有预测性又具有健壮性(对敌方攻击不敏感)的功能，即不能用于区分自然数据和对抗性数据。在几个基准测试上的实验结果表明，该方法对大范围的攻击类型和攻击强度都是有效的。我们的代码可在https://github.com/BashivanLab/afd.获得



## **49. On the Minimal Adversarial Perturbation for Deep Neural Networks with Provable Estimation Error**

具有可证明估计误差的深度神经网络的最小对抗性摄动 cs.LG

Under review on IEEE journal Transactions on Pattern Analysis and  Machine Intelligence

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2201.01235v1)

**Authors**: Fabio Brau, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: Although Deep Neural Networks (DNNs) have shown incredible performance in perceptive and control tasks, several trustworthy issues are still open. One of the most discussed topics is the existence of adversarial perturbations, which has opened an interesting research line on provable techniques capable of quantifying the robustness of a given input. In this regard, the Euclidean distance of the input from the classification boundary denotes a well-proved robustness assessment as the minimal affordable adversarial perturbation. Unfortunately, computing such a distance is highly complex due the non-convex nature of NNs. Despite several methods have been proposed to address this issue, to the best of our knowledge, no provable results have been presented to estimate and bound the error committed. This paper addresses this issue by proposing two lightweight strategies to find the minimal adversarial perturbation. Differently from the state-of-the-art, the proposed approach allows formulating an error estimation theory of the approximate distance with respect to the theoretical one. Finally, a substantial set of experiments is reported to evaluate the performance of the algorithms and support the theoretical findings. The obtained results show that the proposed strategies approximate the theoretical distance for samples close to the classification boundary, leading to provable robustness guarantees against any adversarial attacks.

摘要: 尽管深度神经网络(DNNs)在感知和控制任务中表现出了令人难以置信的性能，但一些值得信赖的问题仍然悬而未决。讨论最多的话题之一是对抗性扰动的存在，这为能够量化给定输入的稳健性的可证明技术开辟了一条有趣的研究路线。在这一点上，输入到分类边界的欧几里德距离表示作为最小负担得起的对抗性扰动的健壮性评估。不幸的是，由于神经网络的非凸性，计算这样的距离非常复杂。尽管已经提出了几种方法来解决这个问题，但据我们所知，还没有提出可证明的结果来估计和限制所犯的错误。本文通过提出两种轻量级策略来解决这一问题，以求最小对抗性扰动。与最新的方法不同，该方法允许建立相对于理论距离的近似距离的误差估计理论。最后，通过大量的实验对算法的性能进行了评估，并支持了理论研究结果。结果表明，对于接近分类边界的样本，所提出的策略近似于理论距离，对任何敌意攻击都具有可证明的鲁棒性保证。



## **50. Adversarial Transformation of Spoofing Attacks for Voice Biometrics**

语音生物特征欺骗攻击的对抗性变换 eess.AS

**SubmitDate**: 2022-01-04    [paper-pdf](http://arxiv.org/pdf/2201.01226v1)

**Authors**: Alejandro Gomez-Alanis, Jose A. Gonzalez-Lopez, Antonio M. Peinado

**Abstracts**: Voice biometric systems based on automatic speaker verification (ASV) are exposed to \textit{spoofing} attacks which may compromise their security. To increase the robustness against such attacks, anti-spoofing or presentation attack detection (PAD) systems have been proposed for the detection of replay, synthesis and voice conversion based attacks. Recently, the scientific community has shown that PAD systems are also vulnerable to adversarial attacks. However, to the best of our knowledge, no previous work have studied the robustness of full voice biometrics systems (ASV + PAD) to these new types of adversarial \textit{spoofing} attacks. In this work, we develop a new adversarial biometrics transformation network (ABTN) which jointly processes the loss of the PAD and ASV systems in order to generate white-box and black-box adversarial \textit{spoofing} attacks. The core idea of this system is to generate adversarial \textit{spoofing} attacks which are able to fool the PAD system without being detected by the ASV system. The experiments were carried out on the ASVspoof 2019 corpus, including both logical access (LA) and physical access (PA) scenarios. The experimental results show that the proposed ABTN clearly outperforms some well-known adversarial techniques in both white-box and black-box attack scenarios.

摘要: 基于自动说话人验证(ASV)的语音生物识别系统容易受到威胁其安全性的攻击。为了提高对此类攻击的鲁棒性，已经提出了用于检测基于重放、合成和语音转换的攻击的反欺骗或呈现攻击检测(PAD)系统。最近，科学界表明，PAD系统也容易受到敌意攻击。然而，据我们所知，还没有研究全语音生物特征识别系统(ASV+PAD)对这些新型的敌意\文本{欺骗}攻击的鲁棒性。在这项工作中，我们开发了一个新的对抗性生物特征转换网络(ABTN)，它联合处理PAD和ASV系统的丢失，以产生白盒和黑盒对抗性文本{欺骗}攻击。该系统的核心思想是生成敌意\文本{欺骗}攻击，这些攻击能够欺骗PAD系统而不被ASV系统检测到。这些实验是在ASVspoof 2019年语料库上进行的，包括逻辑访问(LA)和物理访问(PA)场景。实验结果表明，提出的ABTN在白盒和黑盒攻击场景中的性能都明显优于一些著名的对抗性技术。



