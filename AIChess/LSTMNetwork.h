#pragma once

#include "LSTM.h"
#include <vector>
#include <string>
#include <iostream>

template <typename T>
class LSTMNetwork
{
public:
    // Constructeur : Initialise un réseau avec plusieurs couches LSTM
    LSTMNetwork(const std::vector<size_t>& layer_sizes)
    {
        if (layer_sizes.size() < 2)
        {
            throw std::invalid_argument("Le réseau doit avoir au moins deux couches (entrée et sortie).");
        }

        for (size_t i = 0; i < layer_sizes.size() - 1; ++i)
        {
            layers.emplace_back(LSTM<T>(layer_sizes[i], layer_sizes[i + 1], layer_sizes[i + 1]));
        }
    }

    // Passage avant à travers toutes les couches
    Tensor<T> forward(Tensor<T> input)
    {
        Tensor<T> current_input = input;
        for (auto& lstm : layers)
        {
            current_input = lstm.forward(current_input);
        }
        return current_input;
    }

    // Mutation de toutes les couches
    void mutate(float mutation_rate, float mutation_strength = 0.1)
    {
        for (auto& lstm : layers)
        {
            lstm.mutate(mutation_rate, mutation_strength);
        }
    }

    // Croisement entre deux réseaux pour créer un enfant
    LSTMNetwork breed(const LSTMNetwork& other) const
    {
        if (layers.size() != other.layers.size())
        {
            throw std::invalid_argument("Les réseaux doivent avoir le même nombre de couches pour effectuer un croisement.");
        }

        std::vector<size_t> layer_sizes;
        for (const auto& lstm : layers)
        {
            layer_sizes.push_back(lstm.input_size);
        }
        layer_sizes.push_back(layers.back().output_size);

        LSTMNetwork child(layer_sizes);

        for (size_t i = 0; i < layers.size(); ++i)
        {
            child.layers[i] = layers[i].crossover(other.layers[i]);
        }

        return child;
    }

    // Sauvegarde du réseau dans un fichier
    void save(const std::string& filename) const
    {
        std::ofstream file(filename, std::ios::binary);

        if (!file.is_open())
        {
            throw std::runtime_error("Impossible d'ouvrir le fichier pour sauvegarder le réseau.");
        }

        for (const auto& lstm : layers)
        {
            lstm.saveToStream(file);
        }

        file.close();
    }

    // Chargement du réseau depuis un fichier
    void load(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open())
        {
            throw std::runtime_error("Erreur lors du chargement du fichier.");
        }

        for (auto& lstm : layers)
        {
            lstm.loadFromStream(file);
        }

        file.close();
    }

    // Réinitialisation de la mémoire de toutes les couches
    void resetMemory()
    {
        for (auto& lstm : layers)
        {
            lstm.resetMemory();
        }
    }

    std::vector<LSTM<T>> layers; // Liste des couches LSTM
};

/*
int main() {
    try {
        // === Initialisation des paramètres ===
        std::vector<size_t> layer_sizes = {3, 5, 4}; // Taille des couches

        // Création d'un réseau LSTM avec plusieurs couches
        LSTMNetwork<float> network(layer_sizes);

        // Création d'un tenseur d'entrée (exemple)
        Tensor<float> input({layer_sizes[0], 1}, {1.0f, 2.0f, 3.0f});

        // === Passage avant ===
        Tensor<float> output = network.forward(input);

        std::cout << "Sortie après passage avant : ";
        for (const auto &val : output.data) {
            std::cout << val << " ";
        }
        std::cout << "\n" << std::endl;

        // === Mutation ===
        network.mutate(0.1);   // Mutation avec un taux de mutation de 10%

        network.resetMemory(); // Réinitialisation de la mémoire

        Tensor<float> mutated_output = network.forward(input);

        std::cout << "Sortie après mutation des poids : ";
        for (const auto &val : mutated_output.data) {
            std::cout << val << " ";
        }
        std::cout << "\n" << std::endl;

        // === Sauvegarde du réseau dans un fichier ===
        const std::string filename = "lstm_network.bin";

        network.save(filename);

        std::cout << "Réseau sauvegardé dans le fichier : " << filename << "\n" << std::endl;

        // === Chargement du réseau depuis un fichier ===

         LSTMNetwork<float> loaded_network(layer_sizes);

         loaded_network.load(filename);

         loaded_network.resetMemory();

         Tensor<float> loaded_output = loaded_network.forward(input);

         std::cout << "Sortie après chargement du réseau sauvegardé : ";
         for (const auto &val : loaded_output.data) {
             std::cout << val << " ";
         }
         std::cout << "\n" << std::endl;

         // === Croisement entre deux réseaux ===
         LSTMNetwork<float> child_network = network.breed(loaded_network);

         Tensor<float> child_output = child_network.forward(input);

         std::cout << "Sortie après croisement des réseaux : ";
         for (const auto &val : child_output.data) {
             std::cout << val << " ";
         }
         std::cout << "\n" << std::endl;

     } catch (const std::exception &e) {
         // Gestion des erreurs
         std::cerr << "Erreur rencontrée : " << e.what() << std::endl;
     }

     return 0;
}
*/