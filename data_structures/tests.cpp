#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include "scapegoat_tree.hpp"
typedef int Type;
// Helper function to check if the tree contains all values in a vector
template<typename T>
bool containsAll(const data_structures::scapegoat_tree::ScapeGoatTree<T>& tree, const std::vector<T>& values) {
    for (const auto& val : values) {
        if (!tree.search(val)) return false;
    }
    return true;
}

void testBasicInsertion() {
    std::cout << "Testing Basic Insertion..." << std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree;
    tree.insert(10);
    tree.insert(20);
    tree.insert(5);

    assert(tree.search(10));
    assert(tree.search(20));
    assert(tree.search(5));
    assert(!tree.search(15));
    std::cout << "Basic Insertion Passed!" << std::endl;
}

void testDeletion() {
    std::cout << "Testing Deletion..." << std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree;
    tree.insert(10);
    tree.insert(20);
    tree.insert(5);
    tree.insert(15);
    tree.insert(25);

    // Delete leaf
    assert(tree.deleteValue(25));
    assert(!tree.search(25));

    // Delete node with one child
    assert(tree.deleteValue(20));
    assert(!tree.search(20));
    assert(tree.search(15));

    // Delete node with two children
    tree.insert(20);
    tree.insert(25);
    assert(tree.deleteValue(10));
    assert(!tree.search(10));
    assert(tree.search(15));
    assert(tree.search(5));
    assert(tree.search(20));
    assert(tree.search(25));

    std::cout << "Deletion Passed!" << std::endl;
}

void testRebuilding() {
    std::cout << "Testing Rebuilding (Balancing)..." << std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree;
    // Insert values in increasing order to force imbalance
    for (int i = 1; i <= 10; ++i) {
        tree.insert(i);
    }

    // After 10 insertions, it should have triggered some rebuilds
    // Scapegoat tree with alpha = 2/3 should be relatively balanced
    for (int i = 1; i <= 10; ++i) {
        assert(tree.find_node(i));
    }

    std::string balancedStatus = tree.isBalanced();
    std::cout << "Balance status: " << balancedStatus << std::endl;
    // We expect it to be balanced according to the internal alpha property
    assert(balancedStatus.find("Yes") != std::string::npos || balancedStatus.find("YES") != std::string::npos || balancedStatus.find("is balanced") != std::string::npos);

    std::cout << "Rebuilding Passed!" << std::endl;
}

void testOperators() {
    std::cout << "Testing Operators..." << std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree1;
    tree1.insert(10);
    tree1.insert(20);

    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree2;
    tree2.insert(10);
    tree2.insert(20);

    assert(tree1 == tree2);

    tree2.insert(30);
    assert(tree1 != tree2);

    // Test operator+
    tree1 + 30;
    assert(tree1.search(30));
    assert(tree1 == tree2);

    // Test operator-
    tree1 - 30;
    assert(!tree1.search(30));
    assert(tree1 != tree2);

    // Test operator!
    data_structures::scapegoat_tree::ScapeGoatTree<Type> emptyTree;
    assert(!emptyTree);
    assert(!!tree1);

    std::cout << "Operators Passed!" << std::endl;
}

void testBatchOperations() {
    std::cout << "Testing Batch Operations..." << std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree;
    std::vector<Type> values;
    values.push_back(1);
    values.push_back(2);
    values.push_back(3);
    values.push_back(4);
    values.push_back(5);

    tree.insertBatch(values);
    for (int i = 1; i <= 5; ++i) {
        assert(tree.find_node(i));
    }

    tree.deleteBatch(values);
    for (int i = 1; i <= 5; ++i) {
        assert(!tree.find_node(i));
    }

    std::cout << "Batch Operations Passed!" << std::endl;
}

void testCopyAndMove() {
    std::cout << "Testing Copy and Move..." << std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree1;
    tree1.insert(10);
    tree1.insert(20);
    tree1.insert(30);

    // Copy constructor
    data_structures::scapegoat_tree::ScapeGoatTree tree2(tree1);
    assert(tree2 == tree1);
    assert(tree2.search(10));
    assert(tree2.search(20));
    assert(tree2.search(30));

    // Copy assignment
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree3 = tree1;
    assert(tree3 == tree1);

    // Move constructor
    const data_structures::scapegoat_tree::ScapeGoatTree tree4(std::move(tree2));
    assert(tree4.search(10));
    assert(tree4.search(20));
    assert(tree4.search(30));
    // tree2 should be empty now
    assert(!tree2);

    // Move assignment
    const data_structures::scapegoat_tree::ScapeGoatTree<Type> tree5 = std::move(tree3);
    assert(tree5.search(10));
    assert(!tree3);

    std::cout << "Copy and Move Passed!" << std::endl;
}

void testUandR() {
    std::cout << "Testing Undo and Redo..." << std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree;

    // Test undo insertion
    tree.insert(10);
    assert(tree.search(10));
    tree.undo();
    assert(!tree.search(10));
// Test redo insertion
    tree.insert(10);
    assert(tree.search(10));
    tree.redo();
    assert(tree.search(10));

    // Test undo deletion
    tree.insert(20);
    assert(tree.search(20));
    tree.deleteValue(20);
    assert(!tree.search(20));
    tree.undo();
    assert(tree.search(20));
    // Test redo deletion
    tree.deleteValue(20);
    assert(!tree.search(20));
    tree.redo();
    assert(!tree.search(20));


    // Test batch undo
    std::vector<Type> values;
    values.push_back(1);
    values.push_back(2);
    values.push_back(3);
    tree.insertBatch(values);
    assert(tree.search(1) && tree.search(2) && tree.search(3));
    tree.undo();
    assert(!tree.search(1) && !tree.search(2) && !tree.search(3));

    // Test batch redo
    tree.insertBatch(values);
    assert(tree.search(1) && tree.search(2) && tree.search(3));
    tree.redo();
    assert(tree.search(1) && tree.search(2) && tree.search(3));


    std::cout << "Undo and Redo Passed!" << std::endl;
}
void testNewMethods() {
    std::cout << "Testing New Query Methods (Min, Max, Successor, Kth, Range)..." << std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree;
    std::vector values = {10, 5, 15, 3, 7, 12, 18};
    for (int v : values) tree.insert(v);

    // Min/Max
    assert(tree.getMin() == 3);
    assert(tree.getMax() == 18);

    // Successor
    assert(tree.getSuccessor(7) == 10);
    assert(tree.getSuccessor(12) == 15);
    try { tree.getSuccessor(18); assert(false); } catch (...) {}

    // Kth Smallest (1-indexed)
    assert(tree.kthSmallest(1) == 3);
    assert(tree.kthSmallest(4) == 10);
    assert(tree.kthSmallest(7) == 18);

    // Sum and Values in Range
    assert(tree.sumInRange(5, 12) == (5 + 7 + 10 + 12));

    std::cout << "New Query Methods Passed!" << std::endl;
}

void stressTest() {
    std::cout << "Starting Extreme Stress Test (Large scale randomized operations)..." << std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree;
    std::set<Type> reference_set;

    constexpr int OPERATIONS = 50000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, 100000);
    std::uniform_int_distribution<int> op_dist(0, 2); // 0: Insert, 1: Delete, 2: Search

    for (int i = 0; i < OPERATIONS; ++i) {
        int val = dist(rng);
        int op = op_dist(rng);

        if (op == 0) { // Insert
            tree.insert(val);
            reference_set.insert(val);
        } else if (op == 1) { // Delete
            tree.deleteValue(val);
            reference_set.erase(val);
        } else { // Search
            assert(tree.search(val) == reference_set.contains(val));
        }

        // Periodically check balance and order statistics
        if (i % 10000 == 0 && !reference_set.empty()) {
            assert(tree.getMin() == *reference_set.begin());
            assert(tree.getMax() == *reference_set.rbegin());

            // Check if tree actually respects the height bound
            std::string report = tree.isBalanced();
            assert(report.find("NOT balanced") == std::string::npos);
        }
    }


    std::cout << "Stress Test: Rebuilding verification..." << std::endl;
    std::vector<Type> inorder_vals;

    assert(tree.kthSmallest(1) == *reference_set.begin());
    assert(tree.kthSmallest(reference_set.size()) == *reference_set.rbegin());

    std::cout << "Extreme Stress Test Passed with " << OPERATIONS << " operations!" << std::endl;
}

void testOrderedInsertion() {
    std::cout << "Testing Ordered Insertion (Forcing Scapegoat rebuilds)..." << std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<Type> tree;
    constexpr int N = 2000;
    for (int i = 1; i <= N; ++i) {
        tree.insert(i);
    }

    assert(tree.getMin() == 1);
    assert(tree.getMax() == N);
    assert(tree.isBalanced().find("NOT balanced") == std::string::npos);
    std::cout << "Ordered Insertion Passed!" << std::endl;
}
void testIterator() {
    std::cout << "Testing the Iterator..."<<std::endl;
    data_structures::scapegoat_tree::ScapeGoatTree<int> tree;
   std::vector list = {1647,6332,6104,6967,6278,9555,6216,5658,8067,549,1608,6265,5987,6102,6639,7952,4640,401,5034,7886,9848,5919,632,7589,2051,6348,501,5877,3339,2013,3991,3453,4127,95,386,7432,2613,6970,1067,5619,7592,5497,4791,9270,3704,3523,5134,9375,4256,1128,7811,3327,7960,4582,6914,3848,8880,6233,4398,2553,5460,4807,9726,9417,1072,5554,7101,2299,1159,8897,2353,4034,8019,4156,2309,2657,5309,4648,6776,2781,7957,6415,1469,5206,8549,1739,2309,3377,4560,3354,6961,9209,1198,7497,1105,8122,8892,6781,3794,7198,5889,2813,2351,6989,8362,724,4137,5045,6079,735,5421,4341,1946,7288,5762,8089,3906,2379,2727,1505,5778,3350,630,7653,7663,3642,843,3129,5967,7805,7655,6995,2630,4748,2346,6113,9002,6026,1235,7793,8927,2684,5182,1376,2878,2690,2183,7088,8411,3355,3732,1723,9805,1593,6939,724,3818,6134,6674,2667,7330,5382,1815,899,7559,3448,2275,6061,2199,8004,9941,141,7274,8460,8678,9350,5798,8103,4070,1236,3974,8233,9647,1586,8981,5275,1729,8203,6560,1620,73,2950,2300,6929,3822,8081,5122,5838,6693,9113,2033,494,294,1691,2739,2421,9510,5066,2143,1152,6228,1446,732,4409,4323,9203,6758,5159,2632,3853,9269,8459,5300,1519,5902,9907,5729,2035,1289,9324,7135,8000,6159,1675,5055,2589,3669,4779,4973,98,672,8733,9440,6581,8992,9266,3839,332,5274,4717,2775,3754,797,3328,4621,484,6861,3509,3823,93,1549,454,9331,7949,4951,4562,7210,9914,9467,603,9964,5946,7446,7129,9754,9355,4434,6826,1861,1557,3287,5966,4232,1699,4159,2202,5649,3456,8668,7448,6233,860,9369,2531,1172,1615,2529,5849,9658,6293,9471,6836,2612,5261,1416,5379,2674,9285,240,1013,8490,1324,5011,6197,8620,563,7766,5441,4321,7497,4913,4824,4278,8281,6914,7883,7065,3236,385,6976,3506,4363,9632,9818,9054,7413,2609,9759,4034,9461,6432,7413,6094,4324,8120,407,7098,2784,9108,944,7493,6169,6992,1303,4299,139,9520,592,5416,7064,1648,7325,1229,9865,2040,2858,9656,4812,8908,3788,3386,3249,7529,1220,9201,2330,9929,2817,9611,909,4697,9584,35,8831,4179,7796,8479,627,3928,3538,9491,916,600,7559,9142,4124,2792,2668,1423,556,4271,4634,470,6750,9787,7940,7107,1502,4579,5238,9421,4367,1542,3802,3335,9890,1205,2271,2728,3388,671,4712,5096,5394,5317,7155,2532,2794,4591,9589,8015,8184,8763,8643,5748,5095,4125,1439,5228,622,7579,4756,8738,1707,4855,355,9176,1825,1040,2848,2561,7726,8463,8833,7364,9084,164,3115,404,4235,833,7674,9673,7623,8094,3052,4590,9205,8179,8157,377,3414,4753,9,7621,1655,38,8586,8469,1379,1602,2211,125,2939,6042,5463,7336,4223,7682,2004,5724,6648,9426,8783,3249,8324,8696,9018,300,9689,6648,6271,6897,9918,3606,9913,5576,6259,5561,8383,49,4360,7375,5587,6248,9597,9975,5602,5528,492,8400,904,2275,2384,8981,8325,6130,9515,4636,6914,8329,6124,1156,5033,8378,6991,172,7339,3148,6750,5643,6153,5257,2388,1740,5136,1173,7113,1131,4384,637,1800,2925,2309,1667,2227,5203,5307,1082,7667,3515,8041,2655,6823,6798,4644,4867,795,1119,2727,2509,320,9436,5560,114,415,5656,3416,147,5674,4964,1426,5447,1027,4377,2734,199,9803,2838,2840,1825,2700,2587,7538,5145,2447,9068,8631,864,9610,8386,8283,4736,1593,611,1150,1691,5381,3630,6491,7331,5562,6748,3168,4443,5744,5492,7128,6180,1682,5393,3117,1994,7337,7149,9300,3145,4445,4255,8517,5276,1320,1464,9909,4173,480,2503,3022,3801,5702,1761,6681,6471,7318,9346,1333,752,132,9272,1642,3642,4423,1098,8012,9786,5047,2495,1743,4057,8155,6281,3983,9082,8007,5101,1758,7061,8980,5954,4697,9715,903,7996,8511,3757,7491,880,4792,7573,1324,3782,8495,4693,2455,405,3921,7025,8139,1721,3816,6116,3318,8102,3152,2507,5730,5316,6694,3559,6993,5876,5505,337,9333,7719,224,338,3187,1387,8322,5485,4313,754,4377,2731,8384,8994,7002,2357,531,1514,3041,6583,7383,8175,5407,7891,4456,7496,2775,3201,8770,9476,7206,844,601,8831,759,2598,6756,5803,8984,4175,6619,4718,9638,9595,5099,6200,8946,5895,7121,7156,8322,1229,9739,2855,1690,5877,2432,4437,7285,5800,8398,8771,6311,6082,1608,6971,3003,2332,9666,209,9469,9478,9910,5577,7248,5405,4691,2585,8158,7183,7921,3806,1916,7660,1139,8551,3537,6354,7781,7423,4941,4358,1232,9316,869,6285,6668,2886,887,9617,6790,9335,1623,486,500,3163,9967,2240,1340,4279,6845,4623,3933,8184,9934,7726,8652,9604,63,1211,6990,9522,2060,373,8089,2761,3485,8714,7833,2557,7848,8529,6345,6823,7199,3131,5465,9506,6467,1972,7759,3322,7787,8643,685,7016,5434,1576,7157,6293,4687,6760,9909,1761,9932,2983,4146,6778,374,1087,613,1117,3599,422,8289,1869,9326,7870,2783,6811,1193,1141,8505,696,7037,8695,5745,2607,1681,8596,7035,9977,1439,744,5754,5662,6604,9880,5032,7626,3344,8296,7108,2278,9938,5541,9619,3607,5142,9055,7045,2290,3995,2521,4367,5370,9864,4047,5104,6230,6020,3966,5250,9197,9775,9590,6281,9867,4627,4502,4921,2145,3140,9618,8293,5055,5213,6644,9943,8733,817,9303,1801,5284,4875,3659,7369,669,5036,4631,4595,4860,7418,4398,1015,867,3886,7901,5527,7160,723,9608,2770,5734,8141,4925,1750,9315,7561,165,562,4746,9861,7486,3930,5829,7143,1655,565,8084,5320,8083,5414,5015,820,9839,4796,4776,6538,6170,969,8627,1361,326,3269,2804,7707,8994,2457,7210,7794,9674,3628,8119,7209,2756,6471};
    std::vector<int> mylist;
    for (auto x:list)
        mylist.push_back(x);
    tree.insertBatch(mylist);
    std::vector<int> vect;
    for (auto v : tree)
        vect.push_back(v);
    std::ranges::sort(list);
    auto [first, last] = std::ranges::unique(list);
    list.erase(first, last);
    assert(vect==list);
    std::cout << "Iterator passed"<<std::endl;
}
int main() {

    try {
        testBasicInsertion();
        testDeletion();
        testNewMethods();
        testOrderedInsertion();
        testRebuilding();
        testOperators();
        testBatchOperations();
        testCopyAndMove();
        testUandR();
        stressTest();
        testIterator();
        std::cout << "\n=======================================" << std::endl;
        std::cout << " ALL EXTREME TESTS PASSED SUCCESSFULLY!" << std::endl;
        std::cout << "=======================================" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}