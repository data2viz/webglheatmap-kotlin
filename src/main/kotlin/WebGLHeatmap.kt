package io.data2viz.webglheatmap

import org.khronos.webgl.*
import org.w3c.dom.HTMLCanvasElement
import org.w3c.dom.HTMLImageElement
import kotlin.browser.document

typealias GL = org.khronos.webgl.WebGLRenderingContext

const val vertexShaderBlit = """
    attribute vec4 position;
    varying vec2 texcoord;
    void main(){
        texcoord = position.xy * 0.5 + 0.5;
        gl_Position = position;
    }
    """

const val fragmentShaderBlit = """
    #ifdef GL_FRAGMENT_PRECISION_HIGH
    precision highp int;
    precision highp float;
    #else
    precision mediump int;
    precision mediump float;
    #endif
    uniform sampler2D source;
    varying vec2 texcoord;
    """

class WebGLHeatMap(
    val canvas: HTMLCanvasElement = (document.createElement("canvas") as HTMLCanvasElement).apply {
        this.width = 2
        this.height = 2
    },
    gradientImage: String? = null,
    width: Int? = null,
    height: Int? = null,
    intensityToAlpha: Boolean = false,
    alphaRange: Pair<Float, Float> = Pair(0f, 1f)
) {

    data class ContextAttributes(val antialias: Boolean, val depth: Boolean)
    val arguments = ContextAttributes(false, false)

    val gl: GL = canvas.getContext("experimental-webgl",arguments) as? GL 
            ?: canvas.getContext("webgl", arguments) as? GL
            ?: error("WebGL not supported")

    val quad = gl.createBuffer()
    val gradientTexture: Texture?

    private val shader: Shader
    var width: Int
    var height: Int

    private var heights: Heights

    private val useGradient:Boolean
        get() = gradientTexture != null

    //language=GLSL
    private val colorFunction:String
        get() = if(useGradient)
            """
                uniform sampler2D gradientTexture;
                uniform float intensityDivider;

                vec3 getColor(float intensity){
                    return texture2D(gradientTexture, vec2(intensity, 0.0)).rgb;
                }
                """
        else
            """
                vec3 getColor(float intensity){
                    vec3 blue = vec3(0.0, 0.0, 1.0);
                    vec3 cyan = vec3(0.0, 1.0, 1.0);
                    vec3 green = vec3(0.0, 1.0, 0.0);
                    vec3 yellow = vec3(1.0, 1.0, 0.0);
                    vec3 red = vec3(1.0, 0.0, 0.0);

                    vec3 color = (
                        fade(-0.25, 0.25, intensity)*blue +
                        fade(0.0, 0.5, intensity)*cyan +
                        fade(0.25, 0.75, intensity)*green +
                        fade(0.5, 1.0, intensity)*yellow +
                        smoothstep(0.75, 1.0, intensity)*red
                    );
                    return color;
                }
                """
    init {
        gl.enableVertexAttribArray(0)
        gl.blendFunc(GL.ONE, GL.ONE)
        gradientTexture = loadTexture(gradientImage)

        //language=GLSL
        val outpt = if (intensityToAlpha)
            """
                 vec4 alphaFun(vec3 color, float intensity){
                    float alpha = smoothstep(${alphaRange.first.asDynamic().toFixed(8)},${alphaRange.second.asDynamic().toFixed(8)}, intensity);
                    return vec4(color*alpha, alpha);
                }
                """
        else
            """
                  vec4 alphaFun(vec3 color, float intensity){
                    return vec4(color, 1.0);
                }
                """

        //language=GLSL
        shader = Shader(
            gl, vertexShaderBlit,
            fragmentShaderBlit + """
                float linstep(float low, float high, float value){
                    return clamp((value-low)/(high-low), 0.0, 1.0);
                }

                float fade(float low, float high, float value){
                    float mid = (low+high)*0.5;
                    float range = (high-low)*0.5;
                    float x = 1.0 - clamp(abs(mid-value)/range, 0.0, 1.0);
                    return smoothstep(0.0, 1.0, x);
                }

                $colorFunction

                $outpt

                void main(){
                    float intensity = smoothstep(0.0, 1.0, texture2D(source, texcoord).r);
                    vec3 color = getColor(intensity);
                    gl_FragColor = alphaFun(color, intensity);
                }
                """
        )

        this.width = width ?: if (canvas.offsetWidth > 0) canvas.offsetWidth else 2
        this.height = height ?: if (canvas.offsetHeight > 0) canvas.offsetHeight else 2

        canvas.width = this.width
        canvas.height = this.height

        gl.viewport(0, 0, this.width, this.height)
        gl.bindBuffer(GL.ARRAY_BUFFER, quad)
        val qud = Float32Array(
            arrayOf(
                -1f, -1f, 0f, 1f,
                1f, -1f, 0f, 1f,
                -1f, 1f, 0f, 1f,

                -1f, 1f, 0f, 1f,
                1f, -1f, 0f, 1f,
                1f, 1f, 0f, 1f
            )
        )
        gl.bufferData(GL.ARRAY_BUFFER, qud, GL.STATIC_DRAW)
        gl.bindBuffer(GL.ARRAY_BUFFER, null)
        heights = Heights(this, gl, this.width, this.height)
    }

    private fun loadTexture(gradientImage: String?): Texture?  =
        gradientImage?.let {
            val texture = Texture(gl).apply {
                bind(0)
                size(2, 2)
                nearest()
                clampToEdge()
            }

            val image = js("new Image()") as HTMLImageElement
            image.onload = {
                texture.apply { 
                    bind()
                    upload(image)
                }
            }
            image.src = it
            return@let texture
        }

    fun adjustSize() {
        val canvasWidth = if (canvas.offsetWidth > 0) canvas.offsetWidth else 2
        val canvasHeight = if (canvas.offsetHeight > 0) canvas.offsetHeight else 2
        if (width != canvasWidth || height != canvasHeight){
            gl.viewport(0, 0, canvasWidth, canvasHeight)
            canvas.width = canvasWidth
            canvas.height = canvasHeight
            width = canvasWidth
            height = canvasHeight
            heights.resize(width, height)
        }
    }

    fun display() {
        gl.bindBuffer(GL.ARRAY_BUFFER, quad)
        gl.vertexAttribPointer(0, 4, GL.FLOAT, false, 0, 0)
        heights.nodeFront.bind(0)
//        log("HeatMap.intensityDivider :: $intensityDivider")
        if (gradientTexture != null)
            gradientTexture.bind(1)
        shader.apply {
            use()
            int("source", 0)
            float("intensityDivider", intensityDivider)
            int("gradientTexture", 1)            
        }

        gl.drawArrays(GL.TRIANGLES, 0, 6)
    }

    var intensityDivider = 1f

    fun clear() {
        heights.clear()
        intensityDivider = 1f
    }
    fun update() = heights.update()

    fun clamp(min: Float = 0f, max: Float = 1f) = heights.clamp(min, max)
    fun multiply (value: Float) = heights.multiply(value)
    fun pow (value: Float) = heights.pow(value)
    fun blur() = heights.blur()
    fun addPoint(x: Float, y: Float, size: Float, intensity: Float) {
        heights.addPoint(x, y, size, intensity)
    }
//    fun addPoints(items) = addPoint()

}


class Heights(
    private val heatMap: WebGLHeatMap,
    val gl: GL,
    val width: Int,
    val height: Int) {

    //language=GLSL
    val shader = Shader(
        gl,
        """
                attribute vec4 position, intensity;
                varying vec2 off, dim;
                varying float vIntensity;
                uniform vec2 viewport;

                void main(){
                    dim = abs(position.zw);
                    off = position.zw;
                    vec2 pos = position.xy + position.zw;
                    vIntensity = intensity.x;
                    gl_Position = vec4((pos/viewport)*2.0-1.0, 0.0, 1.0);
                }
        """,
        """
                #ifdef GL_FRAGMENT_PRECISION_HIGH
                    precision highp int;
                    precision highp float;
                #else
                    precision mediump int;
                    precision mediump float;
                #endif
                varying vec2 off, dim;
                varying float vIntensity;
                void main(){
                    float falloff = (1.0 - smoothstep(0.0, 1.0, length(off/dim)));
                    float intensity = falloff*vIntensity;
                    gl_FragColor = vec4(intensity);
                }
        """
    )

    private var nodeBack    = Node(gl, width, height)
    var nodeFront           = Node(gl, width, height)
    private val vertexBuffer = gl.createBuffer()
    private val vertexSize = 8
    private val maxPointCount = 1024 * 10
    private val vertexBufferData = Float32Array(maxPointCount * vertexSize * 6)
    private val vertexBufferViews = Array(maxPointCount) { Float32Array(vertexBufferData.buffer, 0, it * vertexSize * 6) }

    private var maxIntensity = 0f
    var bufferIndex = 0
    var pointCount = 0

    fun resize(width: Int, height: Int) {
        nodeBack.resize(width, height)
        nodeFront.resize(width, height)
    }

    fun update() {
        if (pointCount > 0) {
            gl.enable(GL.BLEND)
            nodeFront.use()
            gl.bindBuffer(GL.ARRAY_BUFFER, vertexBuffer)
            gl.bufferData(GL.ARRAY_BUFFER, vertexBufferViews[pointCount], GL.STREAM_DRAW)

            val positionLoc = shader.attribLocation("position")
            val intensityLoc = shader.attribLocation("intensity")

            gl.enableVertexAttribArray(1)
            gl.vertexAttribPointer(positionLoc, 4, GL.FLOAT, false, 8 * 4, 0 * 4)
            gl.vertexAttribPointer(
                index = intensityLoc,
                size = 4,
                type = GL.FLOAT,
                normalized = false,
                stride = 8 * 4,
                offset = 4 * 4)

            shader.apply { 
                use()
                vec2("viewport", width.toFloat(), height.toFloat())
            }
            gl.drawArrays(GL.TRIANGLES, 0, pointCount * 6)
            gl.disableVertexAttribArray(1)

            pointCount = 0
            bufferIndex = 0
            nodeFront.end()
            gl.disable(GL.BLEND)
        }
    }

    fun clear() {
        nodeFront.use()
        gl.clearColor(0f, 0f, 0f, 1f)
        gl.clear(GL.COLOR_BUFFER_BIT)
        nodeFront.end()
        maxIntensity = 0f
    }

    //language=GLSL
    val clampShader = Shader(
        gl,
        vertexShaderBlit,
        fragmentShaderBlit + """
                uniform float low, high;
                void main(){
                    gl_FragColor = vec4(clamp(texture2D(source, texcoord).rgb, low, high), 1.0);
                }
            """
    )

    fun clamp(min: Float, max: Float) {
        gl.bindBuffer(GL.ARRAY_BUFFER, heatMap.quad)
        gl.vertexAttribPointer(0, 4, GL.FLOAT, false, 0, 0)
        nodeFront.bind(0)
        nodeBack.use()
        clampShader.apply {
            use()
            int("source", 0)
            float("low", min)
            float("high", max)
        }

        gl.drawArrays(GL.TRIANGLES, 0, 6)
    }

    //language=GLSL
    private val powShader = Shader(
        gl,
        vertexShaderBlit,
        fragmentShaderBlit + """
                uniform vec3 value;
                void main(){
                    gl_FragColor = vec4(pow(texture2D(source, texcoord).rgb, value), 1.0);
                }
            """
    )

    fun pow(value: Float) {
        gl.bindBuffer(GL.ARRAY_BUFFER, heatMap.quad)
        gl.vertexAttribPointer(0, 4, GL.FLOAT, false, 0, 0)
        nodeFront.bind(0)
        nodeBack.use()
        powShader.apply {
            use()
            int("source", 0)
            vec3("value", value, value, value)
        }
        gl.drawArrays(GL.TRIANGLES, 0, 6)
        nodeBack.end()
        swap()
    }

    //language=GLSL
    private val multiplyShader = Shader(
        gl,
        vertexShaderBlit,
        fragmentShaderBlit + """
                uniform float value;
                void main(){
                    gl_FragColor = vec4(texture2D(source, texcoord).rgb*value, 1.0);
                }
            """
    )

    fun multiply(value: Float) {
        gl.bindBuffer(GL.ARRAY_BUFFER, heatMap.quad)
        gl.vertexAttribPointer(0, 4, GL.FLOAT, false, 0, 0)
        nodeFront.bind(0)
        nodeBack.use()
        multiplyShader.apply {
            use()
            int("source", 0)
            float("value", value)
        }
            
        gl.drawArrays(GL.TRIANGLES, 0, 6)
        nodeBack.end()
        swap()
    }

    //language=GLSL
    val blurShader = Shader(
        gl,
        vertexShaderBlit,
        fragmentShaderBlit + """
                uniform vec2 viewport;

                void main(){
                    vec4 result = vec4(0.0);
                    for(int x=-1; x<=1; x++){
                        for(int y=-1; y<=1; y++){
                            vec2 off = vec2(x,y)/viewport;
                            //float factor = 1.0 - smoothstep(0.0, 1.5, length(off));
                            float factor = 1.0;
                            result += vec4(texture2D(source, texcoord+off).rgb*factor, factor);
                        }
                    }
                    gl_FragColor = vec4(result.rgb/result.w, 1.0);
                }
            """
    )

    fun blur() {
        gl.bindBuffer(GL.ARRAY_BUFFER, heatMap.quad)
        gl.vertexAttribPointer(0, 4, GL.FLOAT, false, 0, 0)
        nodeFront.bind(0)
        nodeBack.use()
        blurShader.apply {
            use()
            int("source", 0)
            vec2("viewport", width.toFloat(), height.toFloat())
        }
            
        gl.drawArrays(GL.TRIANGLES, 0, 6)
        nodeBack.end()
        swap()
    }

    fun swap() {
        val tmp = nodeFront
        nodeFront = nodeBack
        nodeBack = tmp
    }

    private fun addVertex(x: Float, y: Float, xs: Float, ys: Float, intensity: Float) {
        vertexBufferData[bufferIndex++] = x
        vertexBufferData[bufferIndex++] = y
        vertexBufferData[bufferIndex++] = xs
        vertexBufferData[bufferIndex++] = ys
        vertexBufferData[bufferIndex++] = intensity
        vertexBufferData[bufferIndex++] = intensity
        vertexBufferData[bufferIndex++] = intensity
        vertexBufferData[bufferIndex++] = intensity
    }

    fun addPoint(x: Float, y: Float, size: Float = 50f, intensity: Float = 0.2f) {
        if (pointCount >= maxPointCount - 1)
            update()
        val ny = height - y
        val s = size / 2
        addVertex(x, ny, -s, -s, intensity)
        addVertex(x, ny, +s, -s, intensity)
        addVertex(x, ny, -s, +s, intensity)
        addVertex(x, ny, -s, +s, intensity)
        addVertex(x, ny, +s, -s, intensity)
        addVertex(x, ny, +s, +s, intensity)
        pointCount += 1
    }
}


class Shader(val gl: GL, vertexShader: String, fragmentShader: String) {
    private val program = gl.createProgram()
    private val vs = gl.createShader(GL.VERTEX_SHADER)!!
    private val fs = gl.createShader(GL.FRAGMENT_SHADER)!!
    private val attribCache = mutableMapOf<String, Int>()
    private val uniformCache = mutableMapOf<String, WebGLUniformLocation?>()
    private val valueCache = mutableMapOf<String, Any?>()

    init {
        gl.attachShader(program, vs)
        gl.attachShader(program, fs)
        compileShader(vs, vertexShader)
        compileShader(fs, fragmentShader)
        link()
    }

    fun compileShader(shader: WebGLShader, source: String) {
        gl.shaderSource(shader, source)
        gl.compileShader(shader)
        check(gl.getShaderParameter(shader, GL.COMPILE_STATUS) as Boolean) { 
            "Shader compilation error:: ${gl.getShaderInfoLog(shader)}" 
        }
    }

    fun link() {
        gl.linkProgram(program)
        check(gl.getProgramParameter(program, GL.LINK_STATUS) as Boolean) {
            "Shader link error:: ${gl.getProgramInfoLog(program)}" 
        }
    }

    fun use() = gl.useProgram(program) 
    fun attribLocation(name: String) = attribCache.getOrPut(name) { gl.getAttribLocation(program, name) }
    fun uniformLoc(name: String) =    uniformCache.getOrPut(name) { gl.getUniformLocation(program, name) }

    fun int(name: String, value: Int) {
        val cached = valueCache[name] as Int?
        if (cached != value) {
            valueCache[name] = value
            val loc = uniformLoc(name)
            if (loc != null) {
                gl.uniform1i(loc, value)
            }
        }
    }

    fun float(name: String, value: Float) {
        val cached = valueCache[name] as Float?
        if (cached != value) {
            valueCache[name] = value
            val loc = uniformLoc(name)
            if (loc != null) {
                gl.uniform1f(loc, value)
            }
        }
    }

    fun vec2(name: String, a: Float, b: Float): Unit?               = uniformLoc(name)?.let { gl.uniform2f(it, a, b) }
    fun vec3(name: String, a: Float, b: Float, c: Float)            = uniformLoc(name)?.let { gl.uniform3f(it, a, b, c) }
    fun vec4(name: String, a: Float, b: Float, c: Float, d: Float)  = uniformLoc(name)?.let { gl.uniform4f(it, a, b, c, d) }

}


class Texture(val gl: GL) {

    //todo vérifier
    val channels = GL.RGBA
    val type = GL.UNSIGNED_BYTE

    val chancount = when (channels) {
        GL.RGBA -> 4
        GL.RGB -> 3
        GL.LUMINANCE_ALPHA -> 2
        else -> 1
    }

    val target = GL.TEXTURE_2D
    val handle = gl.createTexture()

    fun destroy() = gl.deleteTexture(handle)
    fun bind(unit: Int = 0)  {
        require(unit < 16) { "Texture unit too large: $unit" }
        gl.activeTexture(GL.TEXTURE0 + unit)
        gl.bindTexture(target, handle)
    }

    private var width: Int = 1
    private var height: Int = 1

    fun size(width: Int, height: Int) {
        this.width = width
        this.height = height
        gl.texImage2D(target, 0, channels, width, height, 0, channels, type, null)
    }

    fun upload(image: HTMLImageElement) {
        width = image.width
        height = image.height
        gl.texImage2D(target, 0, channels, channels, type, image)
    }

    fun linear() {
        gl.texParameteri(target, GL.TEXTURE_MAG_FILTER, GL.LINEAR)
        gl.texParameteri(target, GL.TEXTURE_MIN_FILTER, GL.LINEAR)
    }

    fun nearest() {
        gl.texParameteri(target, GL.TEXTURE_MAG_FILTER, GL.NEAREST)
        gl.texParameteri(target, GL.TEXTURE_MIN_FILTER, GL.NEAREST)
    }

    fun clampToEdge() {
        gl.texParameteri(target, GL.TEXTURE_WRAP_S, GL.CLAMP_TO_EDGE)
        gl.texParameteri(target, GL.TEXTURE_WRAP_T, GL.CLAMP_TO_EDGE)
    }

    fun repeat() {
        gl.texParameteri(target, GL.TEXTURE_WRAP_S, GL.REPEAT)
        gl.texParameteri(target, GL.TEXTURE_WRAP_T, GL.REPEAT)
    }
}

class Node(gl: GL, var width: Int, var height: Int) {

    val texture = Texture(gl).apply {
        bind(0)
        size(width, height)
        nearest()
        clampToEdge()
    }

    val fbo = Framebuffer(gl).apply { 
        bind()
        color(texture)
        unbind()
    }

    fun use() = fbo.bind()
    fun bind(unit: Int) = texture.bind(unit)
    fun end() = fbo.unbind()
    fun resize(width: Int, height: Int) {
        this.width = width
        this.height = height
        texture.apply {
            bind(0)
            size(width, height)
        }
    }
}

class Framebuffer(val gl: GL) {
    val buffer      = gl.createFramebuffer()!!
    
    fun destroy()   = gl.deleteFramebuffer(buffer)
    fun bind()      = gl.bindFramebuffer(GL.FRAMEBUFFER, buffer)
    fun unbind()    = gl.bindFramebuffer(GL.FRAMEBUFFER, null)

    fun check() =
        when (gl.checkFramebufferStatus(GL.FRAMEBUFFER)) {
            GL.FRAMEBUFFER_UNSUPPORTED                      -> error("Framebuffer is unsupported")
            GL.FRAMEBUFFER_INCOMPLETE_ATTACHMENT            -> error("Framebuffer incomplete attachment")
            GL.FRAMEBUFFER_INCOMPLETE_DIMENSIONS            -> error("Framebuffer incomplete dimensions")
            GL.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT    -> error("Framebuffer incomplete missing attachment")
            else -> {}
        }

    fun color(texture: Texture){
        gl.framebufferTexture2D(GL.FRAMEBUFFER, GL.COLOR_ATTACHMENT0, texture.target, texture.handle, 0)
        check()
    }

    fun depth(buffer: WebGLRenderbuffer) =  
        gl.framebufferRenderbuffer(GL.FRAMEBUFFER, GL.DEPTH_ATTACHMENT, GL.RENDERBUFFER, buffer)
}
