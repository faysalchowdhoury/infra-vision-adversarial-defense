A dirty camera lens hurt my traffic-sign classifier almost as much as an adversarial attack.

I trained a CNN on real traffic-sign images (GTSRB, 98.4% accuracy on 12,630 test images), then tested what happens in the field and under attack.

Accuracy:
- Dirty lens (blur, haze, smudges): 82.5%
- Two random stickers on the sign: 76.0%
- An optimised adversarial patch covering 5% of the image: 61.6%
- Digital attack at ε = 8/255: 14.8%

Then I tried standard defences. A median filter brought the patch back to 75.8%, but it made dirty-lens and sticker images worse. No defence helped everything, and the most common real-world problems got worse under the one that helped most against attacks.

An attacker who knew about the filter cut its benefit again, with just 10 steps of optimisation.

I also added an "integrity score" that flags images whose prediction is fragile. It caught about half of the bad images, but also 10% of clean ones, and it flagged strong attacks less often than weak ones, because a strong attack gives a confident wrong answer. It's useful for routing images to human review, not as an alarm.

For infrastructure asset data, I'd add provenance to every decision: the service returns the image hash and model version, so each inventory record can be traced.

Repo: https://github.com/faysalchowdhoury/infra-vision-adversarial-defense

Asset-management folks: how do you QA automated sign or pavement inventories today?

#ComputerVision #AdversarialML #InfrastructureAssetManagement #Cybersecurity #TransportationEngineering #DeepLearning #Python
