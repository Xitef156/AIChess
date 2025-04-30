#include <iostream>
#include <vector>
#include <unordered_map>
#include <functional>
#include <string>
#include <cmath>
#include <iomanip>
#include <random>
#include <thread>
#include <future>
#include <queue>
#include <filesystem>

#include "Tensor.h"
#include "Board.h"
#include "GameManager.h"
#include "Utils.h"
#include "Genetic.h"
#include "LSTMNetwork.h"

// === Déclaration du booléen global pour activer/désactiver les logs ===
bool enableLogging = false;

std::mutex mtx;

// === Fonction utilitaire pour les logs ===
void Log(const std::string& message)
{
	if (enableLogging)
	{
		std::cout << "[LOG] " << message << "\n";
	}
}

// === Fonctions utilitaires ===

// Fonction pour convertir le plateau en tenseur pour le LSTM
Tensor<double> convertBoardToTensor(const Tensor<Cell>& board)
{
	Log("Conversion du plateau en tenseur...");
	std::vector<double> encodedBoard;
	for (const auto& cell : board.data)
	{
		encodedBoard.push_back(cellToDouble(cell));
	}
	Log("Tenseur généré avec succès.");
	return Tensor<double>({ 64, 1 }, encodedBoard);
}

void load(LSTMNetwork<double>& network, std::vector<double> weights, std::vector<size_t>& layer_sizes)
{
	Log("Chargement des poids dans le réseau...");
	for (auto& layer : network.layers)
	{
		layer.Wf.data.assign(weights.begin(), weights.begin() + layer.Wf.data.size());
		weights.erase(weights.begin(), weights.begin() + layer.Wf.data.size());

		layer.Wi.data.assign(weights.begin(), weights.begin() + layer.Wi.data.size());
		weights.erase(weights.begin(), weights.begin() + layer.Wi.data.size());

		layer.Wo.data.assign(weights.begin(), weights.begin() + layer.Wo.data.size());
		weights.erase(weights.begin(), weights.begin() + layer.Wo.data.size());

		layer.Wc.data.assign(weights.begin(), weights.begin() + layer.Wc.data.size());
		weights.erase(weights.begin(), weights.begin() + layer.Wc.data.size());

		layer.bf.data.assign(weights.begin(), weights.begin() + layer.bf.data.size());
		weights.erase(weights.begin(), weights.begin() + layer.bf.data.size());

		layer.bi.data.assign(weights.begin(), weights.begin() + layer.bi.data.size());
		weights.erase(weights.begin(), weights.begin() + layer.bi.data.size());

		layer.bo.data.assign(weights.begin(), weights.begin() + layer.bo.data.size());
		weights.erase(weights.begin(), weights.begin() + layer.bo.data.size());

		layer.bc.data.assign(weights.begin(), weights.end());
	}
	Log("Chargement des poids terminé.");
}

std::vector<double> unload(LSTMNetwork<double>& network) {
	std::vector<double> weights;

	for (const auto& layer : network.layers) {
		// Ajout des poids de chaque matrice et biais de la couche
		weights.insert(weights.end(), layer.Wf.data.begin(), layer.Wf.data.end());
		weights.insert(weights.end(), layer.Wi.data.begin(), layer.Wi.data.end());
		weights.insert(weights.end(), layer.Wo.data.begin(), layer.Wo.data.end());
		weights.insert(weights.end(), layer.Wc.data.begin(), layer.Wc.data.end());
		weights.insert(weights.end(), layer.bf.data.begin(), layer.bf.data.end());
		weights.insert(weights.end(), layer.bi.data.begin(), layer.bi.data.end());
		weights.insert(weights.end(), layer.bo.data.begin(), layer.bo.data.end());
		weights.insert(weights.end(), layer.bc.data.begin(), layer.bc.data.end());
	}

	return weights;
}

// Fonction pour calculer le nombre total de poids nécessaires pour le réseau LSTM
size_t calculateNumWeights(const std::vector<size_t>& layer_sizes)
{
	Log("Calcul du nombre total de poids nécessaires...");
	size_t numWeights = 0;
	for (size_t i = 0; i < layer_sizes.size() - 1; ++i)
	{
		size_t input_size = layer_sizes[i];
		size_t hidden_size = layer_sizes[i + 1];
		numWeights += 4 * (hidden_size * input_size); // Matrices Wf, Wi, Wo, Wc
		numWeights += 4 * hidden_size;                // Vecteurs bf, bi, bo, bc
	}
	Log("Nombre total de poids calculé : " + std::to_string(numWeights));
	return numWeights;
}

// Fonction d'initialisation des poids aléatoires
auto initializerFunc(size_t numWeights) -> std::vector<double>
{
	Log("Initialisation des poids aléatoires...");
	static thread_local std::mt19937 rng(std::random_device{}());
	std::uniform_real_distribution<double> weightDist(-1.0f, 1.0f);

	std::vector<double> weights(numWeights);
	for (auto& weight : weights)
	{
		weight = weightDist(rng);
	}
	Log("Initialisation des poids terminée.");
	return weights;
}

// Fonction de croisement génétique
auto crossoverFunc(std::vector<double> parent1, std::vector<double> parent2) -> std::vector<double>
{
	Log("Croisement génétique...");
	static thread_local std::mt19937 rng(std::random_device{}());
	std::uniform_real_distribution<double> dist(0.0f, 1.0f);

	std::vector<double> childGenome;
	for (size_t i = 0; i < parent1.size(); ++i)
	{
		double alpha = dist(rng); // Poids pour le croisement
		childGenome.push_back(alpha * parent1[i] + (1 - alpha) * parent2[i]);
	}
	Log("Croisement terminé.");
	return childGenome;
}

// Fonction de mutation génétique
auto mutationFunc(std::vector<double>& genome, double generationProgress, size_t noImprovementCounter)
{
	Log("Mutation génétique...");
	static thread_local std::mt19937 rng(std::random_device{}());
	std::uniform_real_distribution<float> mutationDist(-0.1f * (1.0 - generationProgress),
		0.1f * (1.0 - generationProgress));

	for (auto& weight : genome)
	{
		weight += mutationDist(rng);
	}
	Log("Mutation terminée.");
}

// === Setup du jeu classique ===
GameManager setupClassicChess()
{
	Log("Configuration du jeu d'échecs classique...");
	GameManager game({ 8, 8 });
	definePieceMovements(game); // Définir les mouvements des pièces
	placePieces(game);          // Placer les pièces sur le plateau

	// Affectation correcte via lambda pour capturer le GameManager
	game.eventFunction = [&](Tensor<Cell>& board,
		std::unordered_map<Team, std::vector<PieceInstance>>& teams,
		const std::vector<Piece>& globalPieces,
		Team currentTeam,
		size_t turnNumber) {
			chessEventFunction(&game, board, teams, globalPieces, currentTeam, turnNumber);
		};

	// Callback de promotion IA (toujours dame, modifiable)
	game.promotionCallback = [](const PieceInstance&) -> Pieces {
		return 5; // Toujours dame
		};
	defineVictoryCondition(game); // Définir les conditions de victoire

	Log("Configuration terminée.");
	return game;
}

// Fonction fitness pour évaluer les individus

class Roulette {
public:
	Roulette(int max_value) : max_value_(max_value) {
		if (max_value_ < 0) {
			throw std::invalid_argument("max_value doit être supérieur ou égal à 0.");
		}
		reset();
	}

	int get_next() {
		if (available_numbers_.empty()) {
			reset();
		}
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> distrib(0, available_numbers_.size() - 1);
		int index = distrib(gen);
		int result = available_numbers_[index];
		available_numbers_.erase(available_numbers_.begin() + index);
		return result;
	}

	void reset() {
		available_numbers_.clear();
		for (int i = 0; i < max_value_; ++i) {
			available_numbers_.push_back(i);
		}
	}

private:
	int max_value_;
	std::vector<int> available_numbers_;
};

void evaluatePair(std::vector<Individual<std::vector<double>>>& population, int i, int j, std::vector<size_t> layer_sizes)
{
	try
	{
		auto& genome1 = population[i].genome;
		auto& genome2 = population[j].genome;
		LSTMNetwork<double> network1(layer_sizes);
		LSTMNetwork<double> network2(layer_sizes);
		::load(network1, genome1, layer_sizes);
		::load(network2, genome2, layer_sizes);

		GameManager game = setupClassicChess();

		std::unordered_map<Team, double> scores;

		std::unordered_map<Team, size_t> sizes;
		for (const auto& [teamID, pieces] : game.teams)
		{
			sizes[teamID] = pieces.size();
			scores[teamID] = 0.0;
		}

		const double gamma = 1.1;

		size_t currentTurn = 0;

		while (!std::get<1>(game.victoryFunction()))
		{
			// game.displayBoard();
			LSTMNetwork<double>* network;
			// Génération du tenseur représentant le plateau
			Tensor<double> boardTensor = convertBoardToTensor(game.board);
			Log("Tenseur boardTensor généré avec succès. Taille : " + std::to_string(boardTensor.data.size()));

			// Passage du tenseur dans le réseau LSTM
			if (currentTurn % 2 == 0)
				network = &network1;
			else
				network = &network2;
			Tensor<double> moveTensor = network->forward(boardTensor);
			Log("Tenseur moveTensor généré avec succès. Taille : " + std::to_string(moveTensor.data.size()));

			if (moveTensor.data.empty())
			{
				Log("Erreur : moveTensor est vide !");
				throw std::runtime_error("moveTensor est vide !");
			}

			// Débogage des données des tenseurs
			Log("Données de boardTensor :");
			for (size_t i = 0; i < boardTensor.data.size(); ++i)
			{
				Log("boardTensor[" + std::to_string(i) + "] = " + std::to_string(boardTensor.data[i]));
			}

			Log("Données de moveTensor :");
			for (size_t i = 0; i < moveTensor.data.size(); ++i)
			{
				Log("moveTensor[" + std::to_string(i) + "] = " + std::to_string(moveTensor.data[i]));
			}

			// Récupération des mouvements possibles
			auto possibleMoves = game.getMoves();
			Log("Nombre de mouvements possibles : " + std::to_string(possibleMoves.size()));

			if (possibleMoves.empty())
			{
				Log("Erreur : Aucun mouvement possible !");
				throw std::runtime_error("Aucun mouvement possible !");
			}

			// Afficher tous les mouvements possibles
			for (size_t i = 0; i < possibleMoves.size(); ++i)
			{
				auto [from, to] = possibleMoves[i];
				Log("Mouvement " + std::to_string(i) +
					": from=(" +
					std::to_string(from[0]) +
					", " +
					std::to_string(from[1]) +
					") -> to=(" +
					std::to_string(to[0]) +
					", " +
					std::to_string(to[1]) +
					")");
			}

			// Choix d'un mouvement aléatoire parmi les coups possibles
			size_t moveIndex = std::rand() % possibleMoves.size();
			Log("Indice du mouvement choisi : " + std::to_string(moveIndex));

			auto [from, to] = possibleMoves[moveIndex];
			Log("Mouvement choisi par l'IA : (" +
				std::to_string(from[0]) +
				", " +
				std::to_string(from[1]) +
				") -> (" +
				std::to_string(to[0]) +
				", " +
				std::to_string(to[1]) +
				")");

			// Vérification des dimensions des indices en utilisant shape
			if (from.size() != 2 || to.size() != 2)
			{
				Log("Erreur : Indices invalides pour le mouvement !");
				throw std::runtime_error("Indices invalides pour le mouvement !");
			}

			if (from[0] >= game.board.shape[0] || from[1] >= game.board.shape[1] ||
				to[0] >= game.board.shape[0] || to[1] >= game.board.shape[1])
			{
				Log("Erreur : Indices hors limites !");
				throw std::runtime_error("Indices hors limites !");
			}

			// Exécution du mouvement choisi
			if (!game.executeMove(from, to))
			{
				Log("Erreur : Mouvement invalide proposé par l'IA !");
				break;
			}
			double change = std::exp(std::pow(gamma, -std::pow(currentTurn, 2)));
			// Gestion des scores pour toutes les équipes
			for (auto& [team, pieces] : game.teams)
			{
				if (game.teams[team].size() < sizes[team])
				{
					// Cette équipe a perdu une pièce
					scores[team] -= change;
					// Les autres équipes gagnent des points
					for (auto& [otherTeam, _] : game.teams)
					{
						if (otherTeam != team)
							scores[otherTeam] += change;
						else
							scores[otherTeam] -= change;
					}
				}
				sizes[team] = game.teams[team].size();
			}
			currentTurn++;
		}

		auto [messages, finished, winner] = game.victoryFunction();
		/*
			std::cout << currentTurn << std::endl;
			for (const auto &message : messages)
			{
				std::cout << "Message : " << message << "\n";
			};
			std::cout << "Finished : " << finished << std::endl;
			for (auto &[team, pieces] : game.teams)
			{
				std::cout << "Team : " << team << std::endl;
				for (const auto &p : game.teams[team])
				{
					std::cout << p.piece->type << std::endl;
				}
			}*/

		if (!messages.empty() || finished)
		{
			for (const auto& message : messages)
			{
				// std::cout << message << "\n";
			}
			double change = std::exp(std::pow(gamma, -std::pow(currentTurn, 2)));
			if (finished)
			{
				if (winner > 0)
				{
					scores[winner] += 500 * change;
				}
				else if (winner == 0)
				{
					// Match nul, par exemple
					for (auto& [team, score] : scores)
					{
						score += 500 * change;
					}
				}
			}
		}
		// std::cout << i << std::endl;
		population[i].fitness += scores[1] / population.size();
		population[j].fitness += scores[2] / population.size();
		// std::cout << "Score 0 : " << scores[1] << " ; Score 1 : " << scores[2] << std::endl;
	}
	catch (const std::exception& e) {
		std::cerr << "Erreur dans evaluatePair: " << e.what() << std::endl;
	}
	std::lock_guard<std::mutex> lock(mtx);
}

void fitnessFunc(std::vector<Individual<std::vector<double>>>& population, std::vector<size_t> layer_sizes) {
	Log("Évaluation de la population...");

	for (size_t i = 1; i < population.size(); ++i)
	{
		population[i].fitness = population[0].fitness;
	}

	// Déterminer le nombre de cœurs disponibles
	unsigned int numCores = std::thread::hardware_concurrency();
	unsigned int maxThreads = std::max(1u, (unsigned int)(numCores * 0.8));

	std::vector<std::thread> threads;
	std::mutex mutex; // Pour synchroniser l'accès à la population et aux tâches
	std::queue<std::pair<size_t, size_t>> tasks; // File d'attente des tâches (paires d'indices)

	/*
	Roulette roulette(population.size());
	const size_t games = 10;

	for (size_t i = 0; i < games; i++)
	{
		for (size_t j = 0; j < population.size() / 2; ++j) {
			tasks.push({ roulette.get_next(), roulette.get_next() });
		}
	}
	*/
	for (size_t i = 0; i < population.size(); i++)
	{
		for (size_t j = 0; j < population.size(); ++j)
		{
			if(i != j) tasks.push({ i, j });
		}
	}

	// Fonction de travail pour chaque thread
	auto worker = [&](int threadID) {
		while (true) {
			std::pair<size_t, size_t> task;
			{
				std::unique_lock<std::mutex> lock(mutex);
				if (tasks.empty()) {
					break; // Sortir si plus de tâches
				}
				task = tasks.front();
				tasks.pop();
			}
			evaluatePair(population, task.first, task.second, layer_sizes);
		}
		};

	// Lancer les threads
	for (int i = 0; i < maxThreads; ++i) {
		threads.emplace_back(worker, i);
	}

	// Attendre la fin de tous les threads
	for (auto& thread : threads) {
		thread.join();
	}

	Log("Évaluation de la population terminée.");
}

/*
// === Nouvelle version de chessEventFunction ===
void chessEventFunction(
	GameManager* game,
	Tensor<Cell> &board,
	std::unordered_map<Team, std::vector<PieceInstance>> &teams,
	const std::vector<Piece> &globalPieces,
	Team currentTeam,
	size_t turnNumber)
{
	for (auto &[teamID, pieces] : teams)
	{
		for (auto &piece : pieces)
		{
			if (piece.piece->type == 1)
			{
				size_t lastRow = (teamID == 1 ? board.shape[0] - 1 : 0);

				if (piece.position[0] == lastRow)
				{
					Pieces newPieceType = 5; // Dame par défaut
					if (game && game->promotionCallback)
					{
						newPieceType = game->promotionCallback(piece);
					}
					piece.piece =
						&*std::find_if(globalPieces.begin(), globalPieces.end(),
									   [&](const Piece &p) { return p.type == newPieceType; });
					char displayChar = 'q';
					switch (newPieceType)
					{
					case 5: displayChar = 'q'; break;
					case 4: displayChar = 'r'; break;
					case 3: displayChar = 'b'; break;
					case 2: displayChar = 'n'; break;
					}
					piece.displayChar = (teamID == 1 ? tolower(displayChar) : toupper(displayChar));
				}
			}
		}
	}
}
*/

// === Fonction principale ===
int main()
{
	try
	{
		// Configuration du réseau LSTM et de l'algorithme génétique
		Log("Début de la configuration...");

		std::vector<size_t> layer_sizes = { 64, 128, 64 };
		const size_t populationSize = 50;
		const double mutationRate = 0.2;

		size_t numWeights = calculateNumWeights(layer_sizes);

		GameManager newGameTemplate = setupClassicChess();

		const std::string saveFile = "./ai.bin";

		GeneticAlgorithm<std::vector<double>> ga(
			populationSize,
			mutationRate,
			[&](std::vector<Individual<std::vector<double>>>& population) { fitnessFunc(population, layer_sizes); },
			crossoverFunc,
			mutationFunc,
			[&]() { return initializerFunc(numWeights); });

		ga.saveFunction = [&](std::vector<Individual<std::vector<double>>>& population)
			{
				LSTMNetwork<double> network(layer_sizes);
				::load(network, population[0].genome, layer_sizes);
				network.save(saveFile);
			};
		ga.loadFunction = [&]()
			{
				if (std::filesystem::exists(saveFile))
				{
					LSTMNetwork<double> network(layer_sizes);
					network.load(saveFile);
					std::vector<Individual<std::vector<double>>> population(populationSize, Individual<std::vector<double>>(::unload(network), 0.0));
					return population;
				}
				else
				{
					LSTMNetwork<double> network(layer_sizes);
					return std::vector<Individual<std::vector<double>>>(populationSize, Individual<std::vector<double>>(::unload(network), 0.0));
				}
			};

		ga.initializePopulation();
		ga.evolve(5000);

		const auto& bestIndividual = ga.getBestIndividual();

		LSTMNetwork<double> lstmNetwork(layer_sizes);
		load(lstmNetwork, bestIndividual.genome, layer_sizes);

		GameManager game = setupClassicChess();

		while (!std::get<1>(game.victoryFunction()))
		{
			game.displayBoard();

			Tensor<double> boardTensor = convertBoardToTensor(game.board);
			Tensor<double> moveTensor = lstmNetwork.forward(boardTensor);

			auto possibleMoves = game.getMoves();
			if (possibleMoves.empty())
			{
				Log("Aucun mouvement possible !");
				break;
			}

			auto [from, to] = possibleMoves[std::rand() % possibleMoves.size()];
			if (!game.executeMove(from, to))
			{
				Log("Mouvement invalide proposé par l'IA !");
				break;
			}

			game.displayBoard();
		}

		Log("Fin de l'exécution.");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Erreur rencontrée : " << e.what() << "\n";
	}

	return 0;
}
