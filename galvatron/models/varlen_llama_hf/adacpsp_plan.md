1.profile
1.1profile attention(其他agent正在做)
    目标：可以获得一个好的对attention的拟合公式
1.2profile band width
    目标可以给ulysses ring attention usp进行良好的建模
    1.2.1 profile all2all 
        比如64张卡，我们profile all2all size = 8，16，32，64，但是在这里有个问题，
        比如profile size = 8，那我们应该profile 8组 8卡的嘛 profiel size = 32然后profile 2组32卡的嘛
        我主要是为了模拟异构大小的ulysses 通信组，因为可能64张卡，会有一个32卡并行组，两个16卡并行组，我就不知道要怎么profile了，
    1.2.2profile p2p
        为了更好的profile ring attention的带宽，你觉得应该怎么profile
    1.2.3 profile for usp
        比如说usp size = 8 中sp size = 4，cp size = 2怎么profile
            usp size = 16，sp size = 8 cp size = 2，这些应该profile，这里就会有大量的和跨机等需要考虑的事情，

我们的最终目的，是把attention和通信都profile好，可以实际给我们的costmodel去进行使用，但是我不知道怎么整合、利用系统中已经有的，也不知道则怎样的通信profile策略最正确，你认为应该怎么做
        