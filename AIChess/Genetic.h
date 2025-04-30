#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <limits>

// Classe représentant un individu
template <typename T>
class Individual
{
public:
    T genome;
    double fitness;

    Individual(T genome, double fitness = 0.0) : genome(genome), fitness(fitness) {}
};

template <typename T>
class GeneticAlgorithm
{
private:
    std::vector<Individual<T>> population;
    std::function<void(std::vector<Individual<T>>&)> fitnessFunction; // Modifié pour accepter tous les individus
    std::function<T(T, T)> crossoverFunction;
    std::function<void(T&, double, size_t)> mutationFunction;
    std::function<T()> initializer;
    size_t populationSize;
    double mutationRate;

    // Générateur aléatoire thread-safe
    thread_local static std::mt19937 rng;

public:
    std::function<void(std::vector<Individual<T>>&)> saveFunction;
    std::function<std::vector<Individual<T>>()> loadFunction;

    GeneticAlgorithm(size_t populationSize, double mutationRate,
        std::function<void(std::vector<Individual<T>>&)> fitnessFunc, // Modifié ici
        std::function<T(T, T)> crossoverFunc,
        std::function<void(T&, double, size_t)> mutationFunc,
        std::function<T()> initializerFunc)
        : populationSize(populationSize), mutationRate(mutationRate),
        fitnessFunction(fitnessFunc), crossoverFunction(crossoverFunc),
        mutationFunction(mutationFunc), initializer(initializerFunc)
    {
        if (populationSize < 2)
        {
            throw std::invalid_argument("Population size must be at least 2.");
        }
        if (mutationRate < 0.0 || mutationRate > 1.0)
        {
            throw std::invalid_argument("Mutation rate must be between 0 and 1.");
        }
    }

    void initializePopulation()
    {
        for (size_t i = 0; i < populationSize; ++i)
        {
            population.emplace_back(initializer(), 0.0);
        }
        evaluateFitness(); // Évalue immédiatement la fitness après initialisation
    }

    void evaluateFitness()
    {
        // Appelle la fonction de fitness pour toute la population
        fitnessFunction(population);
    }

    void evolve(size_t generations, size_t gamma = 5)
    {
        size_t noImprovementCounter = 0;
        double bestFitness = std::numeric_limits<double>::lowest();
        double currentMutationRate = mutationRate;
        if (loadFunction)
        {
            population = loadFunction();
        }

        for (size_t gen = 0; gen < generations; ++gen)
        {
            evaluateFitness();

            auto bestIt = std::max_element(population.begin(), population.end(),
                [](const Individual<T>& a, const Individual<T>& b) {
                    return a.fitness < b.fitness;
                });
            Individual<T> bestIndividual = *bestIt;

            // Affichage de l'évolution
            if (bestFitness < bestIndividual.fitness)
            {
                if (saveFunction) saveFunction(population);
            }
            std::cout << "Génération " << gen << ": Meilleure fitness = " << bestIndividual.fitness << "\n";

            if (bestIndividual.fitness > bestFitness)
            {
                bestFitness = bestIndividual.fitness;
                noImprovementCounter = 0;
            }
            else
            {
                ++noImprovementCounter;
            }

            std::vector<Individual<T>> newPopulation;

            // Ajout du meilleur individu pour élitisme renforcé
            newPopulation.push_back(bestIndividual);

            // Exploitation : conserver les meilleurs individus
            std::sort(population.begin(), population.end(),
                [](const Individual<T>& a, const Individual<T>& b) {
                    return a.fitness > b.fitness;
                });
            for (size_t i = 1; i < gamma && newPopulation.size() < populationSize; ++i)
            {
                newPopulation.push_back(population[i]);
            }

            if (noImprovementCounter > 20 || calculateFitnessVariance() < 1e-6)
            {
                // Réinitialisation agressive
                size_t numToReset = populationSize / 4; // Réinitialiser 25% de la population
                for (size_t i = 0; i < numToReset; ++i)
                {
                    newPopulation.emplace_back(initializer(), 0.0);
                }

                // Augmenter temporairement le taux de mutation
                currentMutationRate = std::min(currentMutationRate * 2.0, 0.5);
                noImprovementCounter = 0; // Réinitialisation complète du compteur
                std::cout << "Réinitialisation agressive à la génération " << gen << ".\n";
            }

            while (newPopulation.size() < populationSize)
            {
                // Sélection des deux meilleurs parents pour croisement
                Individual<T> parent1 = population[0]; // Meilleur individu
                Individual<T> parent2 = population[1]; // Deuxième meilleur individu

                T childGenome = crossoverFunction(parent1.genome, parent2.genome);
                mutationFunction(childGenome, static_cast<double>(gen) / generations, noImprovementCounter);

                newPopulation.emplace_back(childGenome);
            }

            population = newPopulation;
        }
    }

    double calculateFitnessVariance()
    {
        double meanFitness = 0.0;
        for (const auto& ind : population)
        {
            meanFitness += ind.fitness;
        }
        meanFitness /= population.size();

        double variance = 0.0;
        for (const auto& ind : population)
        {
            variance += std::pow(ind.fitness - meanFitness, 2);
        }
        return variance / population.size();
    }

    const Individual<T>& getBestIndividual() const
    {
        return *std::max_element(population.begin(), population.end(),
            [](const Individual<T>& a, const Individual<T>& b) {
                return a.fitness < b.fitness;
            });
    }
};

//// Définition du générateur aléatoire thread-safe
template <typename T>
thread_local std::mt19937 GeneticAlgorithm<T>::rng(std::random_device{}());

/*
int main()
{
    // Fonction fitness : évalue la qualité de tous les individus simultanément
    auto fitnessFunc = [](std::vector<Individual<std::pair<double, double>>> &population) {
        for (auto &individual : population)
        {
            double x = individual.genome.first;
            double y = individual.genome.second;

            double value = std::pow(8 - (x + y), 10) + -(std::pow(x, y)); // Fonction à minimiser

            if (std::isnan(value))
                individual.fitness = 0;
            else
                individual.fitness = 1.0 / (1.0 + value); // Fitness positive et inversement proportionnelle à la valeur
        }
    };

    // Fonction de croisement : combine deux parents pour produire un enfant
    auto crossoverFunc = [](const std::pair<double, double> &parent1,
                            const std::pair<double, double> &parent2) -> std::pair<double, double> {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        double alpha = dist(rng); // Poids pour le croisement
        double xChild = alpha * parent1.first + (1 - alpha) * parent2.first;
        double yChild = alpha * parent1.second + (1 - alpha) * parent2.second;

        return {xChild, yChild};
    };

    // Fonction de mutation : modifie légèrement un individu
    auto mutationFunc = [](std::pair<double, double> &genome, double generationProgress, size_t noImprovementCounter) {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> mutationDist(-0.1 * (1.0 - generationProgress),
                                                            0.1 * (1.0 - generationProgress));

        genome.first += mutationDist(rng);
        genome.second += mutationDist(rng);
    };

    // Fonction d'initialisation : génère un individu aléatoire
    auto initializerFunc = []() -> std::pair<double, double> {
        static thread_local std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> coordDist(-10.0, 10.0);

        return {coordDist(rng), coordDist(rng)};
    };

    // Création de l'algorithme génétique
    GeneticAlgorithm<std::pair<double, double>> ga(
        500,              // Taille de la population
        0.1,              // Taux de mutation
        fitnessFunc,      // Fonction fitness modifiée pour prendre tous les individus en même temps
        crossoverFunc,    // Fonction de croisement
        mutationFunc,     // Fonction de mutation
        initializerFunc); // Fonction d'initialisation

    ga.initializePopulation(); // Initialisation de la population

    ga.evolve(50000); // Faire évoluer la population pendant 50000 générations

    const auto &best = ga.getBestIndividual();

    // Affichage du meilleur individu trouvé
    std::cout << "Meilleur individu trouvé : (" << best.genome.first << ", " << best.genome.second << ")\n";
    std::cout << "Fitness : " << best.fitness << "\n";

    return 0;
}
*/