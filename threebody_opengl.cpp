#include "planet.hpp"
#include <GLFW/glfw3.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// Window dimensions
const int WIDTH = 1200;
const int HEIGHT = 900;

// Camera settings
float cameraDistance = 3.0f;
float cameraAngleX = 30.0f;
float cameraAngleY = 45.0f;
bool isPaused = false;
float timeScale = 1.0f;

// Mouse control
double lastMouseX = 0, lastMouseY = 0;
bool mousePressed = false;

// Trail storage
struct Trail {
    std::vector<Vector3D> positions;
    int maxLength = 10000;  // Increased for longer trails
    bool unlimited = false;  // If true, trails never fade
    
    void add(Vector3D pos) {
        positions.push_back(pos);
        if (!unlimited && positions.size() > maxLength) {
            positions.erase(positions.begin());
        }
    }
    
    void clear() {
        positions.clear();
    }
};

Trail trail1, trail2, trail3;

// Replay functionality
struct SimulationRecording {
    struct Frame {
        Vector3D pos1, pos2, pos3;
    };
    std::vector<Frame> frames;
    int currentFrame = 0;
    bool isRecording = true;
    bool isReplaying = false;
    
    void record(const Vector3D& p1, const Vector3D& p2, const Vector3D& p3) {
        if (isRecording && !isReplaying) {
            frames.push_back({p1, p2, p3});
        }
    }
    
    void startReplay() {
        if (frames.size() > 0) {
            isReplaying = true;
            isRecording = false;
            currentFrame = 0;
        }
    }
    
    void stopReplay() {
        isReplaying = false;
        isRecording = true;
    }
    
    bool getFrame(Vector3D& p1, Vector3D& p2, Vector3D& p3) {
        if (isReplaying && currentFrame < frames.size()) {
            p1 = frames[currentFrame].pos1;
            p2 = frames[currentFrame].pos2;
            p3 = frames[currentFrame].pos3;
            currentFrame++;
            if (currentFrame >= frames.size()) {
                currentFrame = 0; // Loop replay
            }
            return true;
        }
        return false;
    }
    
    void clear() {
        frames.clear();
        currentFrame = 0;
    }
};

SimulationRecording recording;

static bool parseVec3(const std::string& text, Vector3D& out) {
    std::stringstream ss(text);
    std::string item;
    std::vector<double> parts;
    while (std::getline(ss, item, ',')) {
        try {
            parts.push_back(std::stod(item));
        } catch (...) {
            return false;
        }
    }
    if (parts.size() != 3) {
        return false;
    }
    out = Vector3D(parts[0], parts[1], parts[2]);
    return true;
}

static void printUsage() {
    std::cout
        << "Usage: threebody_opengl [options]\n"
        << "Options:\n"
        << "  --headless               Run simulation without OpenGL and write CSV\n"
        << "  --dt <seconds>           Time step (default: 50.0)\n"
        << "  --steps <count>          Number of steps (default: 2000)\n"
        << "  --out <path>             Output CSV path (default: simulation_data.csv)\n"
        << "  --record <path>          Record CSV during visual mode\n"
        << "  --scale <value>          Visualization position scale (default: 1e-8)\n"
        << "  --m1 <mass> --m2 <mass> --m3 <mass>  Masses in kg\n"
        << "  --p1 x,y,z --p2 x,y,z --p3 x,y,z     Positions in meters\n"
        << "  --v1 x,y,z --v2 x,y,z --v3 x,y,z     Velocities in m/s\n";
}

// Callbacks
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, true);
        if (key == GLFW_KEY_SPACE) isPaused = !isPaused;
        if (key == GLFW_KEY_R) {
            // Clear trails
            trail1.clear();
            trail2.clear();
            trail3.clear();
            recording.clear();
            recording.stopReplay();
            std::cout << "Trails and recording cleared" << std::endl;
        }
        if (key == GLFW_KEY_T) {
            // Toggle unlimited trails
            trail1.unlimited = !trail1.unlimited;
            trail2.unlimited = trail1.unlimited;
            trail3.unlimited = trail1.unlimited;
            std::cout << "Unlimited trails: " << (trail1.unlimited ? "ON" : "OFF") << std::endl;
        }
        if (key == GLFW_KEY_P) {
            // Toggle replay mode
            if (recording.isReplaying) {
                recording.stopReplay();
                isPaused = false;
                std::cout << "Replay stopped" << std::endl;
            } else {
                recording.startReplay();
                isPaused = false;
                std::cout << "Replay started (" << recording.frames.size() << " frames)" << std::endl;
            }
        }
        if (key == GLFW_KEY_EQUAL) timeScale *= 1.2f;  // Speed up
        if (key == GLFW_KEY_MINUS) timeScale /= 1.2f;  // Slow down
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    cameraDistance -= yoffset * 0.2f;
    if (cameraDistance < 0.5f) cameraDistance = 0.5f;
    if (cameraDistance > 10.0f) cameraDistance = 10.0f;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mousePressed = (action == GLFW_PRESS);
        glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (mousePressed) {
        double dx = xpos - lastMouseX;
        double dy = ypos - lastMouseY;
        cameraAngleY += dx * 0.5f;
        cameraAngleX -= dy * 0.5f;
        if (cameraAngleX > 89.0f) cameraAngleX = 89.0f;
        if (cameraAngleX < -89.0f) cameraAngleX = -89.0f;
        lastMouseX = xpos;
        lastMouseY = ypos;
    }
}

// Draw a sphere
void drawSphere(float x, float y, float z, float radius, float r, float g, float b) {
    glColor3f(r, g, b);
    glPushMatrix();
    glTranslatef(x, y, z);
    
    // Draw sphere using triangles
    int slices = 20;
    int stacks = 20;
    
    for (int i = 0; i < stacks; i++) {
        float lat0 = M_PI * (-0.5f + (float)i / stacks);
        float lat1 = M_PI * (-0.5f + (float)(i + 1) / stacks);
        float z0 = sin(lat0);
        float z1 = sin(lat1);
        float zr0 = cos(lat0);
        float zr1 = cos(lat1);
        
        glBegin(GL_QUAD_STRIP);
        for (int j = 0; j <= slices; j++) {
            float lng = 2 * M_PI * (float)j / slices;
            float x = cos(lng);
            float y = sin(lng);
            
            glNormal3f(x * zr0, y * zr0, z0);
            glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius);
            glNormal3f(x * zr1, y * zr1, z1);
            glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius);
        }
        glEnd();
    }
    glPopMatrix();
}

// Draw trail (no fade if unlimited)
void drawTrail(const Trail& trail, float r, float g, float b) {
    if (trail.positions.size() < 2) return;
    
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (size_t i = 0; i < trail.positions.size(); i++) {
        float alpha;
        if (trail.unlimited) {
            // No fade - constant opacity
            alpha = 0.8f;
        } else {
            // Fade effect
            alpha = (float)i / trail.positions.size() * 0.8f;
        }
        glColor4f(r, g, b, alpha);
        glVertex3f(trail.positions[i].getX(), 
                   trail.positions[i].getY(), 
                   trail.positions[i].getZ());
    }
    glEnd();
}

// Draw grid
void drawGrid() {
    glColor3f(0.3f, 0.3f, 0.3f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int i = -10; i <= 10; i++) {
        glVertex3f(i, 0, -10);
        glVertex3f(i, 0, 10);
        glVertex3f(-10, 0, i);
        glVertex3f(10, 0, i);
    }
    glEnd();
}

// Draw axes
void drawAxes() {
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(1.5f, 0, 0);
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 1.5f, 0);
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, 1.5f);
    glEnd();
}

int main(int argc, char** argv) {
    bool headless = false;
    double mass1 = 1e26;
    double mass2 = 1e26;
    double mass3 = 1e26;
    Vector3D p1_pos(1e8, 0, 0);
    Vector3D p2_pos(-5e7, 8.66e7, 0);
    Vector3D p3_pos(-5e7, -8.66e7, 0);
    Vector3D v1_vel(0, 800, 0);
    Vector3D v2_vel(-800 * 0.866, -800 * 0.5, 0);
    Vector3D v3_vel(800 * 0.866, -800 * 0.5, 0);
    double dt = 50.0;
    int numSteps = 2000;
    std::string outPath = "simulation_data.csv";
    bool recordCsv = false;
    std::string recordPath = "simulation_data.csv";
    double scale = 1e-8;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto requireValue = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                printUsage();
                std::exit(1);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--headless") {
            headless = true;
        } else if (arg == "--dt") {
            dt = std::stod(requireValue(arg));
        } else if (arg == "--steps") {
            numSteps = std::stoi(requireValue(arg));
        } else if (arg == "--out") {
            outPath = requireValue(arg);
        } else if (arg == "--record") {
            recordCsv = true;
            recordPath = requireValue(arg);
        } else if (arg == "--scale") {
            scale = std::stod(requireValue(arg));
        } else if (arg == "--m1") {
            mass1 = std::stod(requireValue(arg));
        } else if (arg == "--m2") {
            mass2 = std::stod(requireValue(arg));
        } else if (arg == "--m3") {
            mass3 = std::stod(requireValue(arg));
        } else if (arg == "--p1") {
            if (!parseVec3(requireValue(arg), p1_pos)) {
                std::cerr << "Invalid --p1 value\n";
                return 1;
            }
        } else if (arg == "--p2") {
            if (!parseVec3(requireValue(arg), p2_pos)) {
                std::cerr << "Invalid --p2 value\n";
                return 1;
            }
        } else if (arg == "--p3") {
            if (!parseVec3(requireValue(arg), p3_pos)) {
                std::cerr << "Invalid --p3 value\n";
                return 1;
            }
        } else if (arg == "--v1") {
            if (!parseVec3(requireValue(arg), v1_vel)) {
                std::cerr << "Invalid --v1 value\n";
                return 1;
            }
        } else if (arg == "--v2") {
            if (!parseVec3(requireValue(arg), v2_vel)) {
                std::cerr << "Invalid --v2 value\n";
                return 1;
            }
        } else if (arg == "--v3") {
            if (!parseVec3(requireValue(arg), v3_vel)) {
                std::cerr << "Invalid --v3 value\n";
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage();
            return 1;
        }
    }

    Planet p1("Body1", 1.0, mass1, p1_pos);
    Planet p2("Body2", 1.0, mass2, p2_pos);
    Planet p3("Body3", 1.0, mass3, p3_pos);

    Body body1(p1, Velocity(v1_vel));
    Body body2(p2, Velocity(v2_vel));
    Body body3(p3, Velocity(v3_vel));

    ThreeBodySimulation sim(body1, body2, body3, dt);

    if (headless) {
        std::ofstream outfile(outPath);
        outfile << "time,"
                << "body1_x,body1_y,body1_z,body1_vx,body1_vy,body1_vz,"
                << "body2_x,body2_y,body2_z,body2_vx,body2_vy,body2_vz,"
                << "body3_x,body3_y,body3_z,body3_vx,body3_vy,body3_vz\n";

        for (int i = 0; i < numSteps; i++) {
            sim.step();
            outfile << i * dt << ","
                    << body1.getPlanet().getPosition().getX() << ","
                    << body1.getPlanet().getPosition().getY() << ","
                    << body1.getPlanet().getPosition().getZ() << ","
                    << body1.getVelocity().getVelocity().getX() << ","
                    << body1.getVelocity().getVelocity().getY() << ","
                    << body1.getVelocity().getVelocity().getZ() << ","
                    << body2.getPlanet().getPosition().getX() << ","
                    << body2.getPlanet().getPosition().getY() << ","
                    << body2.getPlanet().getPosition().getZ() << ","
                    << body2.getVelocity().getVelocity().getX() << ","
                    << body2.getVelocity().getVelocity().getY() << ","
                    << body2.getVelocity().getVelocity().getZ() << ","
                    << body3.getPlanet().getPosition().getX() << ","
                    << body3.getPlanet().getPosition().getY() << ","
                    << body3.getPlanet().getPosition().getZ() << ","
                    << body3.getVelocity().getVelocity().getX() << ","
                    << body3.getVelocity().getVelocity().getY() << ","
                    << body3.getVelocity().getVelocity().getZ() << "\n";
        }
        std::cout << "Simulation complete! Data saved to " << outPath << std::endl;
        return 0;
    }

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Create window
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Three-Body Simulation", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    
    // Enable depth testing and blending
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    
    // Set up lighting
    GLfloat light_position[] = { 10.0f, 10.0f, 10.0f, 0.0f };
    GLfloat light_ambient[] = { 0.3f, 0.3f, 0.3f, 1.0f };
    GLfloat light_diffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    
    // Initialize simulation (scaled for visualization)
    double simTime = 0.0;
    std::ofstream recordFile;
    if (recordCsv) {
        recordFile.open(recordPath);
        recordFile << "time,"
                   << "body1_x,body1_y,body1_z,body1_vx,body1_vy,body1_vz,"
                   << "body2_x,body2_y,body2_z,body2_vx,body2_vy,body2_vz,"
                   << "body3_x,body3_y,body3_z,body3_vx,body3_vy,body3_vz\n";
    }
    
    std::cout << "Controls:" << std::endl;
    std::cout << "  Mouse drag: Rotate camera" << std::endl;
    std::cout << "  Mouse scroll: Zoom in/out" << std::endl;
    std::cout << "  SPACE: Pause/Resume" << std::endl;
    std::cout << "  R: Reset trails and recording" << std::endl;
    std::cout << "  T: Toggle unlimited trails (no fade)" << std::endl;
    std::cout << "  P: Play/Stop replay" << std::endl;
    std::cout << "  +/-: Speed up/slow down" << std::endl;
    std::cout << "  ESC: Exit" << std::endl;
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Clear buffers
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Set up projection
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        float aspect = (float)WIDTH / HEIGHT;
        float fov = 45.0f;
        float nearPlane = 0.1f;
        float farPlane = 100.0f;
        float top = nearPlane * tan(fov * M_PI / 360.0);
        glFrustum(-top * aspect, top * aspect, -top, top, nearPlane, farPlane);
        
        // Set up camera
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        float camX = cameraDistance * cos(cameraAngleX * M_PI / 180.0) * cos(cameraAngleY * M_PI / 180.0);
        float camY = cameraDistance * sin(cameraAngleX * M_PI / 180.0);
        float camZ = cameraDistance * cos(cameraAngleX * M_PI / 180.0) * sin(cameraAngleY * M_PI / 180.0);
        
        gluLookAt(camX, camY, camZ, 0, 0, 0, 0, 1, 0);
        
        // Update simulation or replay
        Vector3D pos1, pos2, pos3;
        
        if (recording.isReplaying) {
            // Replay mode - use recorded positions
            if (recording.getFrame(pos1, pos2, pos3)) {
                trail1.add(pos1);
                trail2.add(pos2);
                trail3.add(pos3);
            }
        } else if (!isPaused) {
            // Normal simulation mode
            int stepsThisFrame = (int)(timeScale * 1);
            for (int i = 0; i < stepsThisFrame; i++) {
                sim.step();
            }
            simTime += stepsThisFrame * dt;
            
            // Get current positions
            pos1 = body1.getPlanet().getPosition() * scale;
            pos2 = body2.getPlanet().getPosition() * scale;
            pos3 = body3.getPlanet().getPosition() * scale;

            if (recordCsv) {
                recordFile << simTime << ","
                           << body1.getPlanet().getPosition().getX() << ","
                           << body1.getPlanet().getPosition().getY() << ","
                           << body1.getPlanet().getPosition().getZ() << ","
                           << body1.getVelocity().getVelocity().getX() << ","
                           << body1.getVelocity().getVelocity().getY() << ","
                           << body1.getVelocity().getVelocity().getZ() << ","
                           << body2.getPlanet().getPosition().getX() << ","
                           << body2.getPlanet().getPosition().getY() << ","
                           << body2.getPlanet().getPosition().getZ() << ","
                           << body2.getVelocity().getVelocity().getX() << ","
                           << body2.getVelocity().getVelocity().getY() << ","
                           << body2.getVelocity().getVelocity().getZ() << ","
                           << body3.getPlanet().getPosition().getX() << ","
                           << body3.getPlanet().getPosition().getY() << ","
                           << body3.getPlanet().getPosition().getZ() << ","
                           << body3.getVelocity().getVelocity().getX() << ","
                           << body3.getVelocity().getVelocity().getY() << ","
                           << body3.getVelocity().getVelocity().getZ() << "\n";
            }
            
            // Record for replay
            recording.record(pos1, pos2, pos3);
            
            // Add to trails
            trail1.add(pos1);
            trail2.add(pos2);
            trail3.add(pos3);
        } else {
            // Paused - just get current positions
            pos1 = body1.getPlanet().getPosition() * scale;
            pos2 = body2.getPlanet().getPosition() * scale;
            pos3 = body3.getPlanet().getPosition() * scale;
        }
        
        // Draw scene
        drawGrid();
        drawAxes();
        
        // Draw trails
        glDisable(GL_LIGHTING);
        drawTrail(trail1, 1.0f, 0.3f, 0.3f);
        drawTrail(trail2, 0.3f, 0.3f, 1.0f);
        drawTrail(trail3, 0.3f, 1.0f, 0.3f);
        glEnable(GL_LIGHTING);
        
        // Draw bodies (use positions from simulation or replay)
        drawSphere(pos1.getX(), pos1.getY(), pos1.getZ(), 0.08f, 1.0f, 0.2f, 0.2f);
        drawSphere(pos2.getX(), pos2.getY(), pos2.getZ(), 0.08f, 0.2f, 0.2f, 1.0f);
        drawSphere(pos3.getX(), pos3.getY(), pos3.getZ(), 0.08f, 0.2f, 1.0f, 0.2f);
        
        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    glfwTerminate();
    return 0;
}
