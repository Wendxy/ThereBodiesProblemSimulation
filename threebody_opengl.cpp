#include "planet.hpp"
#include <GLFW/glfw3.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <cmath>
#include <iostream>
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
    int maxLength = 500;
    
    void add(Vector3D pos) {
        positions.push_back(pos);
        if (positions.size() > maxLength) {
            positions.erase(positions.begin());
        }
    }
};

Trail trail1, trail2, trail3;

// Callbacks
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, true);
        if (key == GLFW_KEY_SPACE) isPaused = !isPaused;
        if (key == GLFW_KEY_R) {
            trail1.positions.clear();
            trail2.positions.clear();
            trail3.positions.clear();
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

// Draw trail
void drawTrail(const Trail& trail, float r, float g, float b) {
    if (trail.positions.size() < 2) return;
    
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (size_t i = 0; i < trail.positions.size(); i++) {
        float alpha = (float)i / trail.positions.size();
        glColor4f(r, g, b, alpha * 0.8f);
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

int main() {
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
    double scale = 1e-8;  // Scale positions to fit in view
    double mass = 1e26;
    double radius = 1e8;
    
    Planet p1("Body1", 1.0, mass, Vector3D(radius, 0, 0));
    Planet p2("Body2", 1.0, mass, Vector3D(-radius/2, radius*0.866, 0));
    Planet p3("Body3", 1.0, mass, Vector3D(-radius/2, -radius*0.866, 0));
    
    double v_orbit = 800;
    Body body1(p1, Velocity(Vector3D(0, v_orbit, 0)));
    Body body2(p2, Velocity(Vector3D(-v_orbit*0.866, -v_orbit*0.5, 0)));
    Body body3(p3, Velocity(Vector3D(v_orbit*0.866, -v_orbit*0.5, 0)));
    
    double dt = 50.0;
    ThreeBodySimulation sim(body1, body2, body3, dt);
    
    std::cout << "Controls:" << std::endl;
    std::cout << "  Mouse drag: Rotate camera" << std::endl;
    std::cout << "  Mouse scroll: Zoom in/out" << std::endl;
    std::cout << "  SPACE: Pause/Resume" << std::endl;
    std::cout << "  R: Reset trails" << std::endl;
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
        
        // Update simulation
        if (!isPaused) {
            for (int i = 0; i < (int)(timeScale * 1); i++) {
                sim.step();
            }
            
            // Add to trails
            trail1.add(body1.getPlanet().getPosition() * scale);
            trail2.add(body2.getPlanet().getPosition() * scale);
            trail3.add(body3.getPlanet().getPosition() * scale);
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
        
        // Draw bodies
        Vector3D pos1 = body1.getPlanet().getPosition() * scale;
        Vector3D pos2 = body2.getPlanet().getPosition() * scale;
        Vector3D pos3 = body3.getPlanet().getPosition() * scale;
        
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
