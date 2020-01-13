# Beschreibung
Die Dateien sind im Rahmen der Bachelorarbeit "Einsatzmöglichkeiten von künstlicher Intelligenz in der Bauschadensanierung" an der Hochschule Weihenstephan-Triesdorf entstanden.
Im folgenden sind einige Anmerkungen zu verschiedenen Dateien zu finden.

## 2.1.3.3 Grundlage der Abbildung 6 & 2.1.3.3 Grundlage der Abbildung 7
Im Verlauf der Bachelorarbeit sind einige Convolutional neuronal Networks (CNN) entstanden. Die meisten um verschiedene Netzarchitekturen zu testen und einfach zur Übung.
Zwei davon wurden schließlich verwendet um die beiden Abbildungen 6 und 7 in der Bachelorarbeit zu erzeugen. Die Abbildungen wurden in Excel erstellt.
*Abbildung 6 zeigt wie die Überanpassung in CNNs zu erkennen ist. Das CNN ist daher so ausgelegt, dass eine Überanpassung gut erkennbar auftritt. Die SeparableConv2 Schichten führen eine Faltung mit verschiedenen Fenstergrößen und Schrittweiten aus und führen das Ergebnis dieser Faltungen wieder zusammen ([eine genauere Erläuterung ist hier zu finden:](https://arxiv.org/pdf/1610.02357.pdf)).
*Abbildung 7 zeigt die Verteilungsdichte der Werte der Gewichtungen, um das Ergebnis der Regularisierungsmethoden zu zeigen. In der aktuellen Datei ist dies, die l1 Regularisierung. Die Gewichtungen der untersuchten Schicht werden ab Zeile 120 ausgelesen und in einer csv-Datei hinterlegt. Die Batch Normalization in Zeile 85 hat einen änhlichen Effekt wie die l2 Kernel Regularisierung. Somit sind die Kurve  
