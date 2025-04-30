#pragma once

#include "Tensor.h"
#include <random>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <fstream>
#include <iostream>

template <typename T>
class LSTM
{
public:
    LSTM(size_t input_size, size_t hidden_size, size_t output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size),
        Wf({ hidden_size, input_size }), Wi({ hidden_size, input_size }),
        Wo({ hidden_size, input_size }), Wc({ hidden_size, input_size }),
        bf({ hidden_size, 1 }), bi({ hidden_size, 1 }),
        bo({ hidden_size, 1 }), bc({ hidden_size, 1 }),
        hidden_state({ hidden_size, 1 }), cell_state({ hidden_size, 1 }),
        initial_hidden_state({ hidden_size, 1 }), initial_cell_state({ hidden_size, 1 })
    {
        randomizeWeights();
        validateDimensions();
        saveInitialStates(); // Sauvegarde des états initiaux
    }

    Tensor<T> forward(const Tensor<T>& input)
    {
        validateInput(input);

        // Calcul des portes
        auto forget_gate = applySigmoid(Wf.matmul(input) + bf);
        auto input_gate = applySigmoid(Wi.matmul(input) + bi);
        auto output_gate = applySigmoid(Wo.matmul(input) + bo);
        auto cell_input = applyTanh(Wc.matmul(input) + bc);

        // Mise à jour de l'état cellulaire
        cell_state = forget_gate * cell_state + input_gate * cell_input;

        // Mise à jour de l'état caché
        hidden_state = output_gate * applyTanh(cell_state);

        return hidden_state;
    }

    void mutate(float mutation_rate, float mutation_strength = 0.1)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-mutation_strength, mutation_strength);
        std::uniform_real_distribution<float> rate_dist(0.0f, 1.0f);

        auto mutateFunc = [&dist, &rate_dist, &gen, mutation_rate](T value) -> T {
            return rate_dist(gen) < mutation_rate ? value + dist(gen) : value;
            };

        Wf = Wf.apply(mutateFunc);
        Wi = Wi.apply(mutateFunc);
        Wo = Wo.apply(mutateFunc);
        Wc = Wc.apply(mutateFunc);

        bf = bf.apply(mutateFunc);
        bi = bi.apply(mutateFunc);
        bo = bo.apply(mutateFunc);
        bc = bc.apply(mutateFunc);
    }

    void save(const std::string& filename) const
    {
        std::ofstream file(filename, std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("Impossible d'ouvrir le fichier pour sauvegarder le modèle.");

        Wf.saveToStream(file);
        Wi.saveToStream(file);
        Wo.saveToStream(file);
        Wc.saveToStream(file);

        bf.saveToStream(file);
        bi.saveToStream(file);
        bo.saveToStream(file);
        bc.saveToStream(file);

        hidden_state.saveToStream(file);
        cell_state.saveToStream(file);

        file.close();
    }

    void load(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("Erreur lors du chargement du fichier.");

        Wf.loadFromStream(file);
        Wi.loadFromStream(file);
        Wo.loadFromStream(file);
        Wc.loadFromStream(file);

        bf.loadFromStream(file);
        bi.loadFromStream(file);
        bo.loadFromStream(file);
        bc.loadFromStream(file);

        hidden_state.loadFromStream(file);
        cell_state.loadFromStream(file);

        validateDimensions();
    }

    LSTM crossover(const LSTM& other) const
    {
        if (input_size != other.input_size || hidden_size != other.hidden_size || output_size != other.output_size)
        {
            throw std::invalid_argument("Les dimensions des couches LSTM ne correspondent pas pour effectuer un croisement.");
        }

        LSTM child(input_size, hidden_size, output_size);

        auto crossoverFunc = [](const Tensor<T>& a, const Tensor<T>& b) -> Tensor<T> {
            Tensor<T> result(a.shape);
            for (size_t i = 0; i < a.data.size(); ++i)
            {
                result.data[i] = (i % 2 == 0) ? a.data[i] : b.data[i];
            }
            return result;
            };

        child.Wf = crossoverFunc(Wf, other.Wf);
        child.Wi = crossoverFunc(Wi, other.Wi);
        child.Wo = crossoverFunc(Wo, other.Wo);
        child.Wc = crossoverFunc(Wc, other.Wc);

        child.bf = crossoverFunc(bf, other.bf);
        child.bi = crossoverFunc(bi, other.bi);
        child.bo = crossoverFunc(bo, other.bo);
        child.bc = crossoverFunc(bc, other.bc);

        return child;
    }

    // Sauvegarde des paramètres dans un flux
    void saveToStream(std::ostream& stream) const
    {
        Wf.saveToStream(stream);
        Wi.saveToStream(stream);
        Wo.saveToStream(stream);
        Wc.saveToStream(stream);

        bf.saveToStream(stream);
        bi.saveToStream(stream);
        bo.saveToStream(stream);
        bc.saveToStream(stream);

        hidden_state.saveToStream(stream);
        cell_state.saveToStream(stream);
    }

    // Chargement des paramètres depuis un flux
    void loadFromStream(std::istream& stream)
    {
        Wf.loadFromStream(stream);
        Wi.loadFromStream(stream);
        Wo.loadFromStream(stream);
        Wc.loadFromStream(stream);

        bf.loadFromStream(stream);
        bi.loadFromStream(stream);
        bo.loadFromStream(stream);
        bc.loadFromStream(stream);

        hidden_state.loadFromStream(stream);
        cell_state.loadFromStream(stream);

        validateDimensions();
    }

    void resetMemory()
    {
        hidden_state = initial_hidden_state;
        cell_state = initial_cell_state;
    }

    size_t input_size;
    size_t hidden_size;
    size_t output_size;

    Tensor<T> Wf{}, Wi{}, Wo{}, Wc{};
    Tensor<T> bf{}, bi{}, bo{}, bc{};

    Tensor<T> hidden_state{};
    Tensor<T> cell_state{};

    Tensor<T> initial_hidden_state{};
    Tensor<T> initial_cell_state{};

    static Tensor<T> applySigmoid(const Tensor<T>& x)
    {
        return x.apply([](T value) { return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-value)); });
    }

    static Tensor<T> applyTanh(const Tensor<T>& x)
    {
        return x.apply([](T value) { return static_cast<T>(std::tanh(value)); });
    }

    void validateDimensions() const
    {
        if (Wf.shape != std::vector<size_t>{hidden_size, input_size} ||
            Wi.shape != std::vector<size_t>{hidden_size, input_size} ||
            Wo.shape != std::vector<size_t>{hidden_size, input_size} ||
            Wc.shape != std::vector<size_t>{hidden_size, input_size})
        {
            throw std::invalid_argument("Les dimensions des poids (Wf, Wi, Wo, Wc) ne correspondent pas aux tailles spécifiées.");
        }

        if (bf.shape != std::vector<size_t>{hidden_size, 1} ||
            bi.shape != std::vector<size_t>{hidden_size, 1} ||
            bo.shape != std::vector<size_t>{hidden_size, 1} ||
            bc.shape != std::vector<size_t>{hidden_size, 1})
        {
            throw std::invalid_argument("Les dimensions des biais (bf, bi, bo, bc) ne correspondent pas aux tailles spécifiées.");
        }

        if (hidden_state.shape != std::vector<size_t>{hidden_size, 1} ||
            cell_state.shape != std::vector<size_t>{hidden_size, 1})
        {
            throw std::invalid_argument("Les dimensions des états internes (hidden_state, cell_state) ne correspondent pas aux tailles spécifiées.");
        }
    }

    void validateInput(const Tensor<T>& input) const
    {
        // Vérification de la dimension de l'entrée
        if (input.shape != std::vector<size_t>{input_size, 1})
        {
            throw std::invalid_argument("Dimension de l'entrée invalide. Attendu : [" +
                std::to_string(input_size) + " x 1], reçu : [" +
                std::to_string(input.shape[0]) + " x " +
                std::to_string(input.shape[1]) + "].");
        }
    }

    void randomizeWeights()
    {
        // Initialisation aléatoire des poids et biais avec une distribution uniforme
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dist(-0.1f, 0.1f);

        auto randomFunc = [&dist, &gen](T) -> T { return dist(gen); };

        Wf = Wf.apply(randomFunc);
        Wi = Wi.apply(randomFunc);
        Wo = Wo.apply(randomFunc);
        Wc = Wc.apply(randomFunc);

        bf = bf.apply(randomFunc);
        bi = bi.apply(randomFunc);
        bo = bo.apply(randomFunc);
        bc = bc.apply(randomFunc);

        saveInitialStates(); // Sauvegarde des états initiaux après randomisation
    }

    void saveInitialStates()
    {
        // Sauvegarde des états initiaux pour permettre une réinitialisation future
        initial_hidden_state = hidden_state;
        initial_cell_state = cell_state;
    }
};

/*
int main()
{
    try
    {
        // === Initialisation des paramètres ===
        size_t input_size = 3;   // Taille de l'entrée
        size_t hidden_size = 5;  // Taille de l'état caché
        size_t output_size = 2;  // Taille de la sortie

        // Création d'une couche LSTM
        LSTM<float> lstm(input_size, hidden_size, output_size);

        // Création d'un tenseur d'entrée (exemple)
        Tensor<float> input({input_size, 1}, {1.0f, 2.0f, 3.0f});

        // === Passage avant (Forward pass) ===
        Tensor<float> output = lstm.forward(input);

        std::cout << "Sortie après passage avant : ";
        for (const auto &val : output.data)
        {
            std::cout << val << " ";
        }
        std::cout << "\n" << std::endl;

        // === Mutation des poids et réinitialisation ===
        lstm.mutate(0.1);   // Mutation avec un taux de mutation de 10%
        lstm.resetMemory(); // Réinitialisation de la mémoire

        // Passage avant après mutation
        Tensor<float> mutated_output = lstm.forward(input);

        std::cout << "Sortie après mutation des poids : ";
        for (const auto &val : mutated_output.data)
        {
            std::cout << val << " ";
        }
        std::cout << "\n" << std::endl;

        // === Sauvegarde du modèle dans un fichier ===
        const std::string filename = "lstm_model.bin";
        lstm.save(filename);
        std::cout << "Modèle sauvegardé dans le fichier : " << filename << "\n" << std::endl;

        // === Chargement du modèle depuis un fichier ===
        LSTM<float> loaded_lstm(input_size, hidden_size, output_size);

        loaded_lstm.load(filename);
        loaded_lstm.resetMemory(); // Réinitialisation après chargement

        Tensor<float> loaded_output = loaded_lstm.forward(input);

        std::cout << "Sortie après chargement du modèle sauvegardé : ";
        for (const auto &val : loaded_output.data)
        {
            std::cout << val << " ";
        }
        std::cout << "\n" << std::endl;

    }
    catch (const std::exception &e)
    {
        // Gestion des erreurs
        std::cerr << "Erreur rencontrée : " << e.what() << std::endl;
    }

    return 0;
}
*/